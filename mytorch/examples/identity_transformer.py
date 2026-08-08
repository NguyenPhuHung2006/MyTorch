import mytorch.nn as nn
import mytorch as torch
import mytorch.optim as optim
import numpy as np

# python -m mytorch.examples.identity_transformer

VOCAB_SIZE = 12

BOS_IDX = VOCAB_SIZE - 2
EOS_IDX = VOCAB_SIZE - 1

D_MODEL = 32
NHEAD = 4

NUM_ENCODER_LAYERS = 1
NUM_DECODER_LAYERS = 1

DIM_FEEDFORWARD = 64

SEQ_LEN = 6
BATCH_SIZE = 32

NUM_SAMPLES = 5000
EPOCHS = 30

LR = 1e-3

def generate_dataset(num_samples):
    """
    Generate:

        source = [BOS, tokens..., EOS]

    target = [BOS, tokens..., EOS]
    """

    src = np.random.randint(
        0,
        VOCAB_SIZE - 2,
        size=(num_samples, SEQ_LEN),
    )

    tgt = np.concatenate(
        [
            np.full(
                (num_samples, 1),
                BOS_IDX,
            ),
            src,
            np.full(
                (num_samples, 1),
                EOS_IDX,
            ),
        ],
        axis=1,
    )

    return src, tgt

src_data, tgt_data = generate_dataset(
    NUM_SAMPLES
)

class CopyTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            VOCAB_SIZE,
            D_MODEL,
        )
        
        self.positional_encoding = nn.PositionalEncoding(
            d_model=D_MODEL,
            max_seq_len=SEQ_LEN + 1,
        )

        self.transformer = nn.Transformer(
            d_model=D_MODEL,
            nhead=NHEAD,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_DECODER_LAYERS,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=0.0,
            batch_first=True,
        )

        self.output_projection = nn.Linear(
            D_MODEL,
            VOCAB_SIZE,
        )

    def forward(
        self,
        src,
        tgt,
        tgt_mask=None,
    ):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        output = self.transformer(
            src,
            tgt,
            tgt_mask=tgt_mask,
        )

        output = self.output_projection(output)

        return output


def causal_mask(seq_len):
    return np.triu(
        np.full(
            (seq_len, seq_len),
            -np.inf,
        ),
        k=1,
    )

model = CopyTransformer()

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LR,
)

tgt_mask = causal_mask(SEQ_LEN + 1)

for epoch in range(EPOCHS):

    model.train()

    permutation = np.random.permutation(
        NUM_SAMPLES
    )

    total_loss = 0.0

    for start in range(
        0,
        NUM_SAMPLES,
        BATCH_SIZE,
    ):
        indices = permutation[
            start:start+BATCH_SIZE
        ]

        src = torch.Tensor(src_data[indices])
        tgt = torch.Tensor(tgt_data[indices])

        # ------------------------------------
        # Teacher forcing
        # ------------------------------------

        decoder_input = tgt[:, :-1]

        expected = tgt[:, 1:]

        # ------------------------------------
        # Forward
        # ------------------------------------

        logits = model(
            src,
            decoder_input,
            tgt_mask=tgt_mask,
        )

        # logits:
        # (batch, target_len, vocab_size) -> (N, vocab_size)

        # expected:
        # (batch, target_len) -> (N,)
        
        logits = logits.reshape(-1, VOCAB_SIZE)
        expected = expected.reshape(-1)
        
        loss = criterion(
            logits,
            expected,
        )

        # ------------------------------------
        # Backprop
        # ------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.numpy()

    avg_loss = (
        total_loss
        / (NUM_SAMPLES / BATCH_SIZE)
    )

    print(
        f"Epoch {epoch + 1:02d} "
        f"Loss: {avg_loss:.4f}"
    )
    

def generate(model, src, max_len):
    model.eval()

    # Encode source
    src_emb = model.embedding(src)
    src_emb = model.positional_encoding(src_emb)

    memory = model.transformer.encoder(
        src_emb
    )

    # Start with BOS
    generated = np.array(
        [[BOS_IDX]],
        dtype=np.int64,
    )

    for _ in range(max_len):

        tgt = torch.Tensor(generated)

        tgt_emb = model.embedding(tgt)
        tgt_emb = model.positional_encoding(tgt_emb)

        tgt_mask = causal_mask(
            generated.shape[1]
        )

        decoder_output = model.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
        )

        logits = model.output_projection(
            decoder_output
        )

        # Last timestep
        next_token = np.argmax(
            logits.data[:, -1, :],
            axis=-1,
        )

        generated = np.concatenate(
            [
                generated,
                next_token[:, None],
            ],
            axis=1,
        )

        if next_token[0] == EOS_IDX:
            break

    return generated

src = torch.Tensor(
    np.array([
        [4, 7, 2, 9, 2, 3]
    ])
)

result = generate(
    model,
    src,
    max_len=SEQ_LEN + 1,
)

print(result)