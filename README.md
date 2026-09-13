CEDL

Author: Dian Jiao

CEDL is a hippocampal-inspired language model architecture organized into four
functionally specialized stages: contextual encoding (C), expansion (E), causal
retrieval (D), and context-memory comparison and feedback (L). Two shared
processing sweeps are coupled with a small recurrent probability readout that
combines language predictions with token-successor evidence. Hippocampal pathways
provide the functional inspiration for this design.

PERFORMANCE

The submitted study evaluates CEDL at approximately 45.6 million parameters,
using a 2,048-token context and three seeds on each of two corpora. Each model
receives 536.9 million main language-training targets per corpus, followed by
matched task presentations under a language-retention constraint.

Dataset          Mean NLL    Perplexity    Isolated associative recall
WikiText-103     3.1282      22.83         99.46%
FineWeb-Edu      3.8107      45.18         99.91%

Perplexity is the exponential of mean NLL across seeds. Under the reported
protocol, Mamba-2 obtains perplexities of 24.62 and 48.49, respectively. Public
Transformer and Mamba-2 implementations and a looped-Transformer control provide
the comparisons at approximately 46 million parameters.

The contribution is a language-and-retrieval trade-off: lower language loss and
high isolated lexical recall, with greater training time and scoring latency.
Recall results describe the selected language-preserving systems under this
training recipe; they do not establish the baselines' maximum recall capacity.
Downstream results are mixed, and the biological correspondence remains a
functional analogy.

ACCOMPANYING PAPER

This implementation accompanies the manuscript submitted to Neurocomputing:

Jiao, D. (2026). CEDL: A Hippocampal-Inspired Architecture for Advancing LLMs.
Manuscript submitted to Neurocomputing.

This is a provisional citation to a submitted manuscript, not a published
journal article. Publication details can be added when available.

IMPLEMENTATION

cedl.py contains the standalone model. PyTorch and compatible trained weights
are required; weights are supplied separately. See README.md for installation,
checkpoint loading, usage examples and the verified execution scope.
