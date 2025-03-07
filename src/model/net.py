import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, config, d_model=128, nhead=8, num_encoder_layers=3, num_decoder_layers=3, 
                 dim_feedforward=256, dropout=0.1):
        """
        Initializes the TransformerModel.
        
        Args:
            config: Config object containing parameters (e.g. out_sequence).
            d_model: Dimension of the token embeddings.
            nhead: Number of attention heads.
            num_encoder_layers: Number of encoder layers.
            num_decoder_layers: Number of decoder layers.
            dim_feedforward: Dimension of the feedforward network.
            dropout: Dropout rate.
        """
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.out_sequence = config.out_sequence  # number of output images
        
        # --- CNN Encoder ---
        # A small convolutional network to encode 128x128 single-channel images.
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # -> 16 x 64 x 64
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # -> 32 x 32 x 32
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # -> 32 x 1 x 1
        )
        self.encoder_fc = nn.Linear(32, d_model)
        
        # --- CNN Decoder ---
        # A small network to decode token embeddings into 128x128 images.
        self.decoder_fc = nn.Linear(d_model, 32 * 8 * 8)  # Map to feature map of size (32, 8, 8)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # -> 16 x 16 x 16
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),   # -> 8 x 32 x 32
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1),    # -> 4 x 64 x 64
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1),    # -> 1 x 128 x 128
            nn.Sigmoid()
        )
        
        # --- Transformer ---
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, 
                                          num_encoder_layers=num_encoder_layers, 
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout)
        
        # --- Temporal Encoding ---
        # A linear layer to map the sine of the day difference (a scalar) to the d_model dimension.
        self.temporal_linear = nn.Linear(1, d_model)
        
        # Learned query embeddings for the decoder (one per target output).
        self.decoder_query = nn.Parameter(torch.randn(self.out_sequence, d_model))
    
    def forward(self, input_images, input_temporal, target_temporal=None):
        """
        Forward pass of the model.
        
        Args:
            input_images: Tensor of shape (batch, n1, 1, 128, 128) containing the input images.
            input_temporal: Tensor of shape (batch, n1) containing day differences for input images.
            target_temporal: Tensor of shape (batch, n2) containing day differences for target images.
                             If provided, these are added (after processing) to the learned decoder queries.
        
        Returns:
            output_images: Tensor of shape (batch, n2, 1, 128, 128) representing the generated images.
        """
        batch_size, n1, _, H, W = input_images.shape
        # Flatten batch and sequence dimensions to encode each image individually.
        x = input_images.view(batch_size * n1, 1, H, W)  # -> (batch*n1, 1, 128, 128)
        # (Note: It is assumed that the dataset already selects only the NDSI band, so input is 1 channel.)
        features = self.encoder_cnn(x)  # -> (batch*n1, 32, 1, 1)
        features = features.view(batch_size * n1, 32)
        tokens = self.encoder_fc(features)  # -> (batch*n1, d_model)
        tokens = tokens.view(batch_size, n1, self.d_model)  # -> (batch, n1, d_model)
        
        # Add temporal encoding to input tokens.
        # Compute the sine of the day differences and map to d_model.
        temp_in = torch.sin(input_temporal.unsqueeze(-1))  # (batch, n1, 1)
        temp_in = self.temporal_linear(temp_in)             # (batch, n1, d_model)
        encoder_tokens = tokens + temp_in
        
        # Transformer encoder expects input shape (seq_len, batch, d_model).
        encoder_input = encoder_tokens.transpose(0, 1)  # -> (n1, batch, d_model)
        memory = self.transformer.encoder(encoder_input)  # -> (n1, batch, d_model)
        
        # Prepare the decoder queries.
        # Start with the learned query embeddings (shape: (n2, d_model)) and replicate for the batch.
        decoder_query = self.decoder_query.unsqueeze(1).repeat(1, batch_size, 1)  # -> (n2, batch, d_model)
        if target_temporal is not None:
            # Process target temporal differences: (batch, n2) -> (batch, n2, 1) -> map to (batch, n2, d_model).
            temp_out = torch.sin(target_temporal.unsqueeze(-1))
            temp_out = self.temporal_linear(temp_out)
            # Add the temporal encoding to the learned queries.
            decoder_query = decoder_query + temp_out.transpose(0, 1)
        
        # Transformer decoder: output shape (n2, batch, d_model)
        decoder_output = self.transformer.decoder(decoder_query, memory)
        decoder_output = decoder_output.transpose(0, 1)  # -> (batch, n2, d_model)
        
        # Decode each token to an image.
        y = decoder_output.view(batch_size * self.out_sequence, self.d_model)
        y = self.decoder_fc(y)  # -> (batch*n2, 32*8*8)
        y = y.view(batch_size * self.out_sequence, 32, 8, 8)
        y = self.decoder_conv(y)  # -> (batch*n2, 1, 128, 128)
        output_images = y.view(batch_size, self.out_sequence, 1, 128, 128)
        return output_images
