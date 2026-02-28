"""
Recurrent Autoencoder (RAE) for Temporal Imputation
====================================================

Implements temporally consistent imputation of missing hydrochemical
observations using a GRU-based encoder-decoder architecture.

Key properties
--------------
* Preserves nonlinear temporal sequence dependencies via bidirectional GRU.
* Prevents data leakage — model is fit exclusively on training-year data and
  then applied to all years (including held-out test year).
* Handles variable location histories (not every well appears in all years).
* Gracefully degrades to median imputation when PyTorch is unavailable.

Architecture
------------
Encoder : GRU(n_features → hidden) → Linear(hidden → latent)
Decoder : Linear(latent → hidden) → GRU(latent → hidden) → Linear(hidden → n_features)

Training loss is computed only on *observed* positions (mask=1) so the model
is never penalised for reconstructing values we don't know.  After training,
reconstructed values are spliced in only where the original data was NaN.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional PyTorch imports
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


# ---------------------------------------------------------------------------
# PyTorch model (defined only when torch is available)
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class _RAEModel(nn.Module):
        """
        GRU-based encoder-decoder for short multivariate temporal sequences.

        Parameters
        ----------
        n_features : int
            Number of hydrochemical features.
        hidden_size : int
            GRU hidden state dimension.
        latent_size : int
            Bottleneck (latent code) dimension.
        n_layers : int
            Number of GRU layers in encoder and decoder.
        dropout : float
            Dropout probability (applied between GRU layers when n_layers > 1).
        """

        def __init__(
            self,
            n_features: int,
            hidden_size: int,
            latent_size: int,
            n_layers: int,
            dropout: float,
        ):
            super().__init__()
            self.n_features = n_features
            self.hidden_size = hidden_size
            self.latent_size = latent_size
            self.n_layers = n_layers

            # ---- Encoder ------------------------------------------------
            self.encoder_gru = nn.GRU(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
            self.encoder_fc = nn.Sequential(
                nn.Linear(hidden_size, latent_size),
                nn.Tanh(),
            )

            # ---- Decoder ------------------------------------------------
            self.decoder_init = nn.Linear(latent_size, hidden_size)
            self.decoder_gru = nn.GRU(
                input_size=latent_size,
                hidden_size=hidden_size,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
            self.decoder_fc = nn.Linear(hidden_size, n_features)

        # ------------------------------------------------------------------
        def encode(self, x: "torch.Tensor") -> "torch.Tensor":
            """(batch, seq_len, n_features) → (batch, latent_size)."""
            _, hidden = self.encoder_gru(x)
            h = hidden[-1]                        # last layer: (batch, hidden)
            return self.encoder_fc(h)             # (batch, latent)

        def decode(self, z: "torch.Tensor", seq_len: int) -> "torch.Tensor":
            """(batch, latent) → (batch, seq_len, n_features)."""
            batch_size = z.shape[0]
            decoder_input = z.unsqueeze(1).expand(-1, seq_len, -1)

            h0 = torch.tanh(self.decoder_init(z))          # (batch, hidden)
            h0 = h0.unsqueeze(0).expand(                    # (n_layers, batch, hidden)
                self.n_layers, -1, -1
            ).contiguous()

            out, _ = self.decoder_gru(decoder_input, h0)   # (batch, seq, hidden)
            return self.decoder_fc(out)                     # (batch, seq, n_features)

        def forward(
            self, x: "torch.Tensor"
        ) -> "Tuple[torch.Tensor, torch.Tensor]":
            z = self.encode(x)
            return self.decode(z, x.shape[1]), z


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class RAEImputer:
    """
    Recurrent Autoencoder Imputer for multi-year hydrochemical data.

    Builds temporal sequences per monitoring location across survey years,
    trains a GRU encoder-decoder on available (non-missing) observations,
    then uses the learned representation to fill missing values in a
    temporally consistent manner.

    Leakage prevention
    ------------------
    * ``fit()`` accepts only *training* year data (default: all years
      except the last).
    * ``transform()`` applies the frozen model to any set of years.

    Parameters
    ----------
    hidden_size : int
        GRU hidden state size (default 64).
    latent_size : int
        Bottleneck dimension (default 32).
    n_layers : int
        Number of GRU layers (default 1).
    dropout : float
        Dropout between GRU layers when n_layers > 1 (default 0.1).
    n_epochs : int
        Maximum training epochs (default 200).
    lr : float
        Adam learning rate (default 1e-3).
    batch_size : int
        Mini-batch size during training (default 32).
    patience : int
        Early-stopping patience in epochs (default 30).
    random_state : int
        RNG seed for reproducibility (default 42).
    device : str or None
        ``'cuda'`` / ``'cpu'``; auto-detected when None.
    """

    def __init__(
        self,
        hidden_size: int = 64,
        latent_size: int = 32,
        n_layers: int = 1,
        dropout: float = 0.1,
        n_epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 32,
        patience: int = 30,
        random_state: int = 42,
        device: Optional[str] = None,
    ):
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state

        self._model = None
        self._feature_cols: Optional[List[str]] = None
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None
        self._is_fitted: bool = False
        self._train_years: Optional[List[int]] = None

        if device is None:
            self.device = (
                "cuda"
                if (TORCH_AVAILABLE and torch.cuda.is_available())
                else "cpu"
            )
        else:
            self.device = device

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        data_dict: Dict[int, pd.DataFrame],
        location_keys: List[str],
        feature_cols: List[str],
        train_years: Optional[List[int]] = None,
    ) -> "RAEImputer":
        """
        Fit the RAE on temporal sequences from training years only.

        Parameters
        ----------
        data_dict : dict
            ``{year: DataFrame}`` for all available years.
        location_keys : list of str
            Columns that uniquely identify a monitoring location
            (e.g. ``['district', 'mandal', 'village']``).
        feature_cols : list of str
            Numeric hydrochemical feature columns to impute.
        train_years : list of int, optional
            Years used for fitting.  Defaults to all years except the
            last (i.e. held-out test year).

        Returns
        -------
        self
        """
        if train_years is None:
            years_sorted = sorted(data_dict.keys())
            train_years = years_sorted[:-1]

        self._train_years = train_years
        self._feature_cols = feature_cols

        if not TORCH_AVAILABLE:
            logger.warning(
                "PyTorch not available — RAEImputer falling back to "
                "training-median imputation."
            )
            self._fit_fallback(data_dict, feature_cols, train_years)
            return self

        # Compute normalisation statistics on training data only
        train_frames = [data_dict[y] for y in train_years if y in data_dict]
        if not train_frames:
            logger.warning("No training frames for RAE — using fallback.")
            self._fit_fallback(data_dict, feature_cols, train_years)
            return self

        combined = pd.concat(train_frames, ignore_index=True)
        vals = combined[
            [c for c in feature_cols if c in combined.columns]
        ].apply(pd.to_numeric, errors="coerce")

        self._feature_means = vals.mean().reindex(feature_cols).fillna(0).values.astype(np.float32)
        stds = vals.std().reindex(feature_cols).fillna(1.0).values.astype(np.float32)
        stds[stds < 1e-6] = 1.0
        self._feature_stds = stds

        logger.info(
            f"Fitting RAE on years {train_years} | "
            f"{len(feature_cols)} features | device={self.device}"
        )

        # Build training sequences
        sequences, masks = self._build_sequences(
            data_dict, location_keys, feature_cols, train_years
        )

        if len(sequences) < 2:
            logger.warning(
                f"Only {len(sequences)} temporal sequences — "
                "falling back to median imputation."
            )
            self._fit_fallback(data_dict, feature_cols, train_years)
            return self

        logger.info(
            f"Built {len(sequences)} temporal sequences "
            f"(seq_len={sequences.shape[1]})"
        )

        seq_norm = self._normalize(sequences)

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        n_features = len(feature_cols)
        seq_len = sequences.shape[1]

        self._model = _RAEModel(
            n_features=n_features,
            hidden_size=self.hidden_size,
            latent_size=self.latent_size,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self.device)

        self._train_model(seq_norm, masks, seq_len)
        self._is_fitted = True
        logger.info("RAE fitting complete.")
        return self

    def transform(
        self,
        data_dict: Dict[int, pd.DataFrame],
        location_keys: List[str],
        feature_cols: Optional[List[str]] = None,
        all_years: Optional[List[int]] = None,
    ) -> Dict[int, pd.DataFrame]:
        """
        Impute missing values using the fitted RAE.

        Parameters
        ----------
        data_dict : dict
            ``{year: DataFrame}`` to impute (may include test year).
        location_keys : list of str
            Same columns used during ``fit()``.
        feature_cols : list of str, optional
            Defaults to the columns used during ``fit()``.
        all_years : list of int, optional
            Defaults to all keys in ``data_dict``.

        Returns
        -------
        dict
            ``{year: DataFrame}`` with NaN feature values replaced by
            RAE-reconstructed estimates.
        """
        if feature_cols is None:
            feature_cols = self._feature_cols
        if all_years is None:
            all_years = sorted(data_dict.keys())

        if not self._is_fitted or (not TORCH_AVAILABLE and self._model is None):
            return self._transform_fallback(data_dict, feature_cols)

        if not TORCH_AVAILABLE or self._model is None:
            return self._transform_fallback(data_dict, feature_cols)

        sequences, masks = self._build_sequences(
            data_dict, location_keys, feature_cols, all_years
        )
        loc_ids = self._get_location_ids(data_dict, location_keys, all_years)

        if len(sequences) == 0:
            return {y: data_dict[y].copy() for y in all_years if y in data_dict}

        seq_norm = self._normalize(sequences)
        reconstructed_norm = self._reconstruct(seq_norm)
        reconstructed = self._denormalize(reconstructed_norm)

        # Splice: keep observed values, fill NaN positions with reconstruction
        imputed = sequences.copy()
        nan_mask = ~masks.astype(bool)      # True where value was missing
        imputed[nan_mask] = reconstructed[nan_mask]

        result = self._rebuild_dataframes(
            imputed, masks, loc_ids,
            data_dict, location_keys, feature_cols, all_years,
        )

        n_imputed = int(nan_mask.sum())
        logger.info(
            f"RAE imputed {n_imputed} missing values across "
            f"{len(all_years)} year(s)."
        )
        return result

    def fit_transform(
        self,
        data_dict: Dict[int, pd.DataFrame],
        location_keys: List[str],
        feature_cols: List[str],
        train_years: Optional[List[int]] = None,
    ) -> Dict[int, pd.DataFrame]:
        """Fit on training years then impute all years."""
        self.fit(data_dict, location_keys, feature_cols, train_years)
        return self.transform(data_dict, location_keys, feature_cols)

    # ------------------------------------------------------------------
    # Private helpers — sequence construction
    # ------------------------------------------------------------------

    def _build_sequences(
        self,
        data_dict: Dict[int, pd.DataFrame],
        location_keys: List[str],
        feature_cols: List[str],
        years: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build (n_locs, seq_len, n_features) arrays.

        Returns
        -------
        sequences : np.ndarray  float32
            Feature values; NaN positions filled with training mean.
        masks : np.ndarray  float32
            1 where value was originally observed, 0 where it was NaN
            or the location did not appear in that year.
        """
        years = sorted(years)
        n_years = len(years)
        n_features = len(feature_cols)

        # Collect feature vectors per location per year
        loc_data: Dict[tuple, Dict[int, np.ndarray]] = {}
        loc_order: List[tuple] = []

        for year in years:
            if year not in data_dict:
                continue
            df = data_dict[year]
            avail_loc = [c for c in location_keys if c in df.columns]

            for _, row in df.iterrows():
                loc_id = tuple(
                    str(row[k]).upper().strip() for k in avail_loc
                )
                if loc_id not in loc_data:
                    loc_data[loc_id] = {}
                    loc_order.append(loc_id)

                feat = np.array(
                    [pd.to_numeric(row.get(c, np.nan), errors="coerce")
                     for c in feature_cols],
                    dtype=np.float32,
                )
                loc_data[loc_id][year] = feat

        n_locs = len(loc_order)
        sequences = np.zeros((n_locs, n_years, n_features), dtype=np.float32)
        masks = np.zeros((n_locs, n_years, n_features), dtype=np.float32)

        fill = self._feature_means if self._feature_means is not None else np.zeros(n_features, dtype=np.float32)

        for i, loc_id in enumerate(loc_order):
            for j, year in enumerate(years):
                if year in loc_data.get(loc_id, {}):
                    feat = loc_data[loc_id][year]
                    valid = ~np.isnan(feat)
                    sequences[i, j, valid] = feat[valid]
                    sequences[i, j, ~valid] = fill[~valid]
                    masks[i, j, :] = valid.astype(np.float32)
                else:
                    # Location not present this year → fill with mean, mask=0
                    sequences[i, j, :] = fill

        return sequences, masks

    def _get_location_ids(
        self,
        data_dict: Dict[int, pd.DataFrame],
        location_keys: List[str],
        years: List[int],
    ) -> List[tuple]:
        """Return ordered list of unique location tuples across years."""
        seen = set()
        ordered: List[tuple] = []
        for year in sorted(years):
            if year not in data_dict:
                continue
            df = data_dict[year]
            avail = [c for c in location_keys if c in df.columns]
            for _, row in df.iterrows():
                loc_id = tuple(str(row[k]).upper().strip() for k in avail)
                if loc_id not in seen:
                    seen.add(loc_id)
                    ordered.append(loc_id)
        return ordered

    # ------------------------------------------------------------------
    # Private helpers — normalisation
    # ------------------------------------------------------------------

    def _normalize(self, sequences: np.ndarray) -> np.ndarray:
        if self._feature_means is None:
            return sequences
        return (sequences - self._feature_means) / self._feature_stds

    def _denormalize(self, sequences: np.ndarray) -> np.ndarray:
        if self._feature_means is None:
            return sequences
        return sequences * self._feature_stds + self._feature_means

    # ------------------------------------------------------------------
    # Private helpers — training
    # ------------------------------------------------------------------

    def _train_model(
        self,
        seq_norm: np.ndarray,
        masks: np.ndarray,
        seq_len: int,
    ) -> None:
        X_t = torch.tensor(seq_norm, dtype=torch.float32)
        M_t = torch.tensor(masks, dtype=torch.float32)

        dataset = TensorDataset(X_t, M_t)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        optimizer = optim.Adam(
            self._model.parameters(), lr=self.lr, weight_decay=1e-5
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5, verbose=False
        )

        best_loss = float("inf")
        patience_counter = 0

        self._model.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for X_batch, M_batch in loader:
                X_batch = X_batch.to(self.device)
                M_batch = M_batch.to(self.device)

                optimizer.zero_grad()
                reconstructed, z = self._model(X_batch)

                # Masked reconstruction loss (observed positions only)
                diff = (reconstructed - X_batch) ** 2
                n_obs = M_batch.sum().clamp(min=1.0)
                recon_loss = (diff * M_batch).sum() / n_obs

                # Latent regularisation
                reg_loss = 1e-3 * z.pow(2).mean()

                loss = recon_loss + reg_loss
                loss.backward()
                nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            scheduler.step(avg_loss)

            if avg_loss < best_loss - 1e-6:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                logger.info(
                    f"RAE early stopping at epoch {epoch + 1} "
                    f"(loss={avg_loss:.6f})"
                )
                break

            if (epoch + 1) % 50 == 0:
                logger.debug(
                    f"RAE epoch {epoch + 1}/{self.n_epochs}: "
                    f"loss={avg_loss:.6f}"
                )

        self._model.eval()
        logger.info(f"RAE training done — best loss: {best_loss:.6f}")

    # ------------------------------------------------------------------
    # Private helpers — reconstruction
    # ------------------------------------------------------------------

    def _reconstruct(self, seq_norm: np.ndarray) -> np.ndarray:
        """Run encoder-decoder in batches; return reconstructed array."""
        self._model.eval()
        chunks = []
        with torch.no_grad():
            for i in range(0, len(seq_norm), self.batch_size):
                batch = torch.tensor(
                    seq_norm[i: i + self.batch_size], dtype=torch.float32
                ).to(self.device)
                rec, _ = self._model(batch)
                chunks.append(rec.cpu().numpy())
        return np.concatenate(chunks, axis=0)

    # ------------------------------------------------------------------
    # Private helpers — DataFrame reconstruction
    # ------------------------------------------------------------------

    def _rebuild_dataframes(
        self,
        imputed: np.ndarray,
        masks: np.ndarray,
        loc_ids: List[tuple],
        data_dict: Dict[int, pd.DataFrame],
        location_keys: List[str],
        feature_cols: List[str],
        years: List[int],
    ) -> Dict[int, pd.DataFrame]:
        """
        Write imputed values back into copies of the original DataFrames.

        Only positions that were originally NaN (mask=0) are updated;
        observed values are left untouched.
        """
        years = sorted(years)
        loc_idx: Dict[tuple, int] = {lid: i for i, lid in enumerate(loc_ids)}

        result = {
            year: data_dict[year].copy()
            for year in years
            if year in data_dict
        }

        for y_idx, year in enumerate(years):
            if year not in result:
                continue
            df = result[year]
            avail_loc = [c for c in location_keys if c in df.columns]

            for row_i, row in df.iterrows():
                loc_id = tuple(
                    str(row[k]).upper().strip() for k in avail_loc
                )
                if loc_id not in loc_idx:
                    continue
                seq_i = loc_idx[loc_id]

                for f_idx, feat_col in enumerate(feature_cols):
                    if feat_col not in df.columns:
                        continue
                    was_missing = (
                        pd.isna(row[feat_col])
                        or masks[seq_i, y_idx, f_idx] == 0
                    )
                    if was_missing:
                        df.at[row_i, feat_col] = float(
                            imputed[seq_i, y_idx, f_idx]
                        )

        return result

    # ------------------------------------------------------------------
    # Fallback (no PyTorch)
    # ------------------------------------------------------------------

    def _fit_fallback(
        self,
        data_dict: Dict[int, pd.DataFrame],
        feature_cols: List[str],
        train_years: Optional[List[int]],
    ) -> None:
        """Compute training medians as imputation fallback."""
        train_years = train_years or sorted(data_dict.keys())
        frames = [data_dict[y] for y in train_years if y in data_dict]
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            avail = [c for c in feature_cols if c in combined.columns]
            vals = combined[avail].apply(pd.to_numeric, errors="coerce")
            med = vals.median().reindex(feature_cols).fillna(0).values.astype(np.float32)
            self._feature_means = med
            self._feature_stds = np.ones(len(feature_cols), dtype=np.float32)
        self._is_fitted = True

    def _transform_fallback(
        self,
        data_dict: Dict[int, pd.DataFrame],
        feature_cols: List[str],
    ) -> Dict[int, pd.DataFrame]:
        """Fill NaN with training medians (fallback)."""
        result = {}
        for year, df in data_dict.items():
            df_copy = df.copy()
            if self._feature_means is not None:
                for i, col in enumerate(feature_cols):
                    if col in df_copy.columns:
                        df_copy[col] = df_copy[col].fillna(
                            float(self._feature_means[i])
                        )
            result[year] = df_copy
        return result
