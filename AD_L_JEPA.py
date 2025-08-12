import tensorflow as tf

class ADLJEPA(tf.keras.Model):
    def __init__(self, bev_shape, embed_dim=128, encoder_depth=4, predictor_depth=3):
        super().__init__()
        self.bev_shape = bev_shape  # (H, W, C)
        self.embed_dim = embed_dim

        # Learnable tokens (for empty and masked grids)
        self.empty_token = tf.Variable(tf.random.normal([embed_dim]), trainable=True, name="empty_token")
        self.mask_token = tf.Variable(tf.random.normal([embed_dim]), trainable=True, name="mask_token")

        # Context Encoder (f_θ): series of conv layers to embed context BEV
        self.context_encoder = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
            *[tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu') for _ in range(encoder_depth-1)],
            tf.keras.layers.Conv2D(embed_dim, 1, padding='same', activation=None),
            tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1)),  # Per-grid L2 normalization
        ])

        # Predictor (g_ϕ): small 3-layer convnet
        self.predictor = tf.keras.Sequential([
            *[tf.keras.layers.Conv2D(embed_dim, 3, padding='same', activation='relu') for _ in range(predictor_depth-1)],
            tf.keras.layers.Conv2D(embed_dim, 1, padding='same', activation=None),
            tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1)),  # Per-grid L2 normalization
        ])

        # Target Encoder (f_θ̄): will be maintained via EMA outside this class

    def call(self, context_bev, training=False):
        """
        Args:
            context_bev: BEV tensor with shape (batch, H, W, C), 
                         where grids are replaced with mask/empty tokens as needed.
        Returns:
            context_embeddings: (batch, H, W, embed_dim)
            predicted_embeddings: (batch, H, W, embed_dim)
        """
        context_embeddings = self.context_encoder(context_bev)
        predicted_embeddings = self.predictor(context_embeddings)
        return context_embeddings, predicted_embeddings

    def apply_token(self, bev_grid, mask=None, empty=None):
        """
        Replace grids in BEV tensor with mask or empty tokens.
        Args:
            bev_grid: input BEV grid (batch, H, W, C)
            mask: boolean mask, same shape as BEV spatial dims, True = masked
            empty: boolean mask, same shape as BEV spatial dims, True = empty
        Returns:
            processed BEV grid (with tokens inserted)
        """
        output = bev_grid
        if mask is not None:
            output = tf.where(mask[..., None], self.mask_token, output)
        if empty is not None:
            output = tf.where(empty[..., None], self.empty_token, output)
        return output
