# Experiencing myself on a Transformer for the first time

To gain more insight how to implements the final structure of Fo_Nu i decided to gain more experiences with Transformers and to learn the core structure so this project is a sandBox for FoNu_NLP_TG
# LEARNING SANDBOX FOR FoNu_NLP_TG

What is FoNu_NLP_TG.

FoNu_NLP_TG ("Fo Nu" means "speak" in Ewe, and TG stands for Togo) is a research project focused on experimenting, exploring, and fine-tuning transformers, with a special emphasis on applications for Togolese languages.


## Project Blog

We've started a blog to document our progress and share insights about transformer models and NLP. The blog is available in multiple formats:

- [GitHub Pages](https://lemniscate-world.github.io/FoNu_NLP_TG/) (automatically updated)
- [Source files](blog/) in the repository
- Selected posts on [Medium](https://medium.com/) (coming soon)

## Transformer Architecture Standard

1. Encoder: N layers (usually 6) with self-attention and feed-forward networks.
2. Decoder: N layers with self-attention, source-attention (to encoder), and feed-forward networks.
3. Attention: Mechanism to weigh word importance.
4. Forward Pass: Input → Encoder → Memory → Decoder → Output.

## Methods

Standard: Encoder-Decoder with multi-head attention. (Harvard)
Variants: BERT (encoder-only), GPT (decoder-only).
Customization: You can adjust N, hidden size, or attention heads, but the structure is usually fixed.

## Attention Mechanism
- How It Works: Attention calculates "scores" between words. For "Hello world", it checks how much "Hello" relates to "world" using their hidden states.
- Training: The model learns these relationships from data (e.g., "Hello" often precedes "world").
- Multi-Head Attention: Looks at multiple relationships at once (e.g., syntax, meaning).

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Trans.git
cd Izzy-nlpV1

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy models (if needed)
python -m spacy download en_core_web_sm
```

## Project Structure
i will just list the most important class/structure
- `Izzy-nlpV1/`: Implementation based on the original paper but more lighter( i think)
  - `transformer.py`: Core transformer 
  - `encoder.py`: The Encoder class
  - `decoder.py`: The Decoder class
  - `positionalEncoding.py`: The class to calculate the position and do the embeddings
  - `multiHead.py`: The class that do the multiHeadAttention mechanism


## Requirements

See `requirements.txt` for the complete list.

## Papers

- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/#prelims)
![Transformer Architecture](https://nlp.seas.harvard.edu/images/the_transformer_architecture.jpg)
- [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)
- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)


More to come ...
