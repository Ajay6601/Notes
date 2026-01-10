# Tokenization Deep Dive - Complete Fundamental Notes

**Version 1.0 - Comprehensive Guide**  
**Sources**: HuggingFace Tokenizers, SentencePiece paper, OpenAI tiktoken, Karpathy's minbpe, Google's WordPiece, Meta's Llama tokenizer, Production cases from GPT-4, Llama 3, Claude

---

## Table of Contents
1. [Why Tokenization Matters](#1-why-tokenization-matters)
2. [Tokenization Methods](#2-tokenization-methods)
3. [Tokenizer Training](#3-tokenizer-training)
4. [Special Tokens](#4-special-tokens)
5. [Tokenization Challenges](#5-tokenization-challenges)
6. [Tokenization Impact](#6-tokenization-impact)
7. [Production Case Studies](#7-production-case-studies)
8. [Interview Questions](#8-interview-questions)

---

## 1. Why Tokenization Matters

### 1.1 The Core Problem

**Neural networks operate on numbers, not text.** Tokenization bridges this gap by converting text into discrete units (tokens) that can be represented as integers.

**Key Question**: Why not use characters or words directly?

**Character-level issues**:
- Sequences become extremely long (100-200x longer)
- Harder to learn meaningful patterns
- More computation needed

**Word-level issues**:
- Vocabulary explosion (millions of words)
- Out-of-vocabulary (OOV) problem
- Can't handle misspellings or rare words
- Language-specific (English has ~170,000 words)

**Subword tokenization (BPE, WordPiece) solves both**:
- Finite vocabulary (30K-100K tokens)
- Can represent any text (via subwords)
- Balances sequence length and vocabulary size

### 1.2 Fundamental Concepts

**Token**: The smallest unit of text the model processes
- Can be: character, subword, word, or even bytes

**Vocabulary**: The set of all possible tokens
- GPT-2: 50,257 tokens
- GPT-4: ~100,000 tokens
- Llama 3: 128,256 tokens

**Tokenization process**:
```
Text → Pre-tokenization → Token splitting → Token IDs → Model input
"Hello world" → ["Hello", " world"] → [15496, 995] → Embeddings
```

**Critical insight from Andrej Karpathy**:
> "Tokenization is a completely separate stage of pre-processing. It has its own training set (different from the LLM training set), and therefore its own train/test split. LLMs don't see characters - they see tokens."

---

## 2. Tokenization Methods

### 2.1 Byte Pair Encoding (BPE)

**Invented**: 1994 (for compression), adapted for NLP in 2016

**Core idea**: Start with characters, iteratively merge the most frequent pair.

**Algorithm**:
1. Initialize vocabulary with all bytes/characters
2. Find most frequent adjacent pair in corpus
3. Merge this pair, add to vocabulary
4. Repeat until desired vocabulary size

**Example**:
```
Text: "low low low lower lowest"

Initial: l-o-w l-o-w l-o-w l-o-w-e-r l-o-w-e-s-t

Step 1: Most frequent pair = "lo" (appears 5 times)
Merge: lo-w lo-w lo-w lo-w-e-r lo-w-e-s-t
Add "lo" to vocab

Step 2: Most frequent pair = "low" (appears 5 times)
Merge: low low low low-e-r low-e-s-t
Add "low" to vocab

Step 3: Most frequent pair = "lowe"
Merge: low low low lowe-r lowe-s-t
Add "lowe" to vocab

Final vocab: {l, o, w, e, r, s, t, lo, low, lowe, ...}
```

**Used by**: GPT-2, GPT-3, GPT-4 (tiktoken), RoBERTa, BART

**Advantages**:
- Simple and effective
- Data-driven (learns from corpus)
- Handles rare words via subwords
- Deterministic encoding

**Disadvantages**:
- Greedy algorithm (not optimal)
- Sensitive to corpus statistics
- Different tokenization for train vs test data (if vocabulary changes)

**Production example (OpenAI tiktoken)**:
```python
import tiktoken

enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
tokens = enc.encode("Hello, world!")
print(tokens)  # [9906, 11, 1917, 0]
print(enc.decode(tokens))  # "Hello, world!"
```

### 2.2 Byte-Level BPE

**Innovation**: Operate on bytes (0-255) instead of characters

**Key advantage**: Can tokenize any text (any language, any character)
- No unknown tokens (UNK) ever
- UTF-8 text is just bytes

**Used by**: GPT-2, GPT-3, GPT-4, Llama 2/3, Mistral

**How it works**:
1. Encode text as UTF-8 bytes
2. Apply BPE on byte sequences
3. Base vocabulary = 256 bytes (guaranteed)

**Example**:
```
"Hello" in UTF-8:
H = 72, e = 101, l = 108, l = 108, o = 111
Bytes: [72, 101, 108, 108, 111]

BPE merges:
[72, 101] → Token 256 ("He")
[108, 108] → Token 257 ("ll")
Final: [256, 257, 111] = ["He", "ll", "o"]
```

**Why OpenAI uses byte-level BPE** (from GPT-2 paper):
- "Universal tokenization" - works for any language
- No preprocessing needed (no lowercasing, no accent stripping)
- Robust to noise, typos, emoji, code

### 2.3 WordPiece (BERT Tokenization)

**Developed by**: Google (for BERT)

**Key difference from BPE**: Uses likelihood-based scoring instead of frequency

**Algorithm**:
1. Start with character vocabulary
2. Score each merge by likelihood increase
3. Choose merge that maximizes training data likelihood
4. Repeat until vocabulary size reached

**Scoring function**:
```
Score(merge) = log P(corpus with merge) / log P(corpus without merge)
```

**Used by**: BERT, DistilBERT, ELECTRA, ALBERT

**Special prefix**: `##` indicates subword continuation
```
"playing" → ["play", "##ing"]
"player" → ["play", "##er"]
```

**Advantages over BPE**:
- More principled (likelihood-based)
- Better handles morphology

**Disadvantages**:
- Slower training
- More complex implementation

### 2.4 SentencePiece (Language-Agnostic)

**Developed by**: Google (2018)

**Key innovation**: Treats text as raw byte stream (no pre-tokenization)

**Why it matters**:
- Language-agnostic (works for Chinese, Japanese, Thai with no spaces)
- Reversible (can decode back to exact original text)
- Handles whitespace as special character `▁` (U+2581)

**Used by**: T5, mT5, XLM-RoBERTa, Llama, Mistral, most multilingual models

**Example**:
```
Input: "Hello world"
SentencePiece: ["▁Hello", "▁world"]

Input: "こんにちは" (Japanese)
SentencePiece: ["▁こん", "にち", "は"]
```

**Two algorithms supported**:
1. **BPE mode**: Similar to standard BPE
2. **Unigram LM mode**: Probabilistic approach (see next)

**Key insight from T5 paper**:
> "SentencePiece removes the need for language-specific preprocessing, making it ideal for multilingual models."

### 2.5 Unigram Language Model

**Approach**: Start with large vocabulary, iteratively prune

**Algorithm**:
1. Start with huge vocabulary (all possible substrings)
2. For each token, compute loss if removed
3. Remove tokens with smallest loss impact
4. Repeat until target vocabulary size

**Probabilistic tokenization**:
- Multiple tokenizations possible
- Choose highest probability segmentation
- Can sample during training (regularization)

**Example**:
```
"unbelievable" could be:
- ["un", "believ", "able"] (probability 0.6)
- ["un", "believe", "able"] (probability 0.3)
- ["unbe", "liev", "able"] (probability 0.1)

Choose highest probability: ["un", "believ", "able"]
```

**Used by**: ALBERT, T5 (with SentencePiece), XLNet

**Advantages**:
- More flexible than BPE
- Better for morphologically rich languages
- Can use sampling during training

**Disadvantages**:
- Slower inference (need to compute probabilities)
- More complex training

### 2.6 Word-Level Tokenization

**Simplest approach**: Split on whitespace/punctuation

**Example**:
```
"Hello, world!" → ["Hello", ",", "world", "!"]
```

**Problems**:
- Vocabulary size explodes (170K+ words in English)
- OOV problem (can't handle unseen words)
- Language-specific rules needed
- Can't handle typos, rare words

**Rarely used in modern LLMs** - only for small-scale experiments

### 2.7 Character-Level Tokenization

**Ultra-simple**: Each character is a token

**Example**:
```
"Hello" → ["H", "e", "l", "l", "o"]
```

**Advantages**:
- Tiny vocabulary (26 letters + punctuation)
- No OOV problem
- Language-agnostic

**Disadvantages**:
- 5-10x longer sequences
- Harder to learn long-range dependencies
- Much slower training/inference

**Used by**: ByT5 (Google), CANINE (character-level BERT)

**When to use**: Very low-resource languages, noisy text (OCR, social media)

### 2.8 Code Tokenization vs Natural Language

**Key differences**:

**Natural language**:
- Whitespace is meaningful (word boundaries)
- Punctuation separates sentences
- Case matters for proper nouns

**Code**:
- Indentation is critical (Python)
- Syntax characters (`;`, `{`, `}`, `()`) must be separate tokens
- Variable names should be split intelligently (`getUserName` → `get`, `User`, `Name`)
- Comments and strings have different semantics

**Codex/GPT-4 code tokenization** (from OpenAI):
- Separate vocabulary for code tokens
- Camel case splitting: `getUserName` → `["get", "User", "Name"]`
- Preserve indentation: 4 spaces = single token
- Special handling for common patterns (`==`, `!=`, `++`)

**Example**:
```python
# Natural language tokenizer:
"getUserName" → ["getUser", "Name"]  # Not ideal

# Code-aware tokenizer:
"getUserName" → ["get", "User", "Name"]  # Better
"def hello():" → ["def", "hello", "(", ")", ":"]  # Syntax preserved
```

**Llama 3 approach** (128K vocabulary):
- Unified tokenizer for text + code
- More code-specific tokens (parentheses, brackets, operators)
- 5-10% efficiency gain on code vs Llama 2

---

## 3. Tokenizer Training

### 3.1 Training from Scratch

**Steps**:
1. **Corpus selection**: Collect representative text
2. **Pre-tokenization**: Split into words/whitespace
3. **Vocabulary initialization**: Start with base (characters or bytes)
4. **Iterative merging**: Apply BPE/WordPiece algorithm
5. **Vocabulary saving**: Store merge rules and token IDs

**Code example (conceptual)**:
```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

# Initialize tokenizer
tokenizer = Tokenizer(BPE())

# Configure trainer
trainer = BpeTrainer(
    vocab_size=30000,
    min_frequency=2,
    special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
)

# Train on corpus
files = ["corpus1.txt", "corpus2.txt"]
tokenizer.train(files, trainer)

# Save
tokenizer.save("my_tokenizer.json")
```

### 3.2 Vocabulary Construction

**Key decisions**:

**Vocabulary size**:
- Small (10K-30K): Fast, but longer sequences
- Medium (30K-50K): Balanced (GPT-2)
- Large (50K-100K): Shorter sequences, slower (GPT-4)
- Very large (100K+): Multimodal or multilingual (Llama 3: 128K)

**Rule of thumb** (from Karpathy):
> "Vocabulary size is a hyperparameter. Bigger isn't always better. GPT-2's 50K was well-chosen for English."

**Frequency threshold**:
- Min frequency = 2: Include pairs that appear at least twice
- Too low: Noise in vocabulary
- Too high: Miss rare but useful merges

**Example from GPT-2 training**:
```
Corpus: 40GB of web text
Vocabulary: 50,257 tokens
Training time: ~1 hour on single GPU
Result: ~80% compression (5 chars → 1 token avg)
```

### 3.3 Merge Operations in BPE

**Merge rule**: Pair of tokens → New token

**Example progression**:
```
Iteration 1:
Corpus: "a b c a b c a b"
Most frequent pair: ("a", "b") appears 3 times
Merge: Add "ab" to vocabulary
Result: "ab c ab c ab"

Iteration 2:
Most frequent pair: ("ab", "c") appears 3 times
Merge: Add "abc" to vocabulary
Result: "abc abc abc"
```

**Stored as merge table**:
```
Merge #1: ("a", "b") → "ab"
Merge #2: ("ab", "c") → "abc"
...
```

**Inference**: Apply merges in order
```
Input: "a b c"
Apply merge #1: "ab c"
Apply merge #2: "abc"
Final: ["abc"]
```

### 3.4 Training Corpus Selection

**Critical decision**: Corpus determines vocabulary

**GPT-2 corpus** (WebText):
- 40GB of text from Reddit links
- Upvoted content (quality filter)
- Diverse topics
- Result: English-centric vocabulary

**Llama 3 corpus**:
- 15 trillion tokens (30x larger than Llama 2)
- Multilingual (100+ languages)
- Code (GitHub, StackOverflow)
- Result: 128K vocabulary for better coverage

**Key insight from Chinchilla paper**:
> "Training data quality matters more than quantity. 1GB of high-quality text > 10GB of noisy text."

**Best practices**:
1. **Diversity**: Cover all domains (news, books, code, conversation)
2. **Quality**: Filter low-quality text (ads, spam, gibberish)
3. **Balance**: Equal representation (avoid bias toward common phrases)
4. **Size**: 10GB minimum, 100GB+ ideal

### 3.5 Pre-tokenization Steps

**Before applying BPE/WordPiece**:

**Common steps**:
1. **Whitespace splitting**: "Hello world" → ["Hello", "world"]
2. **Punctuation handling**: "Hello!" → ["Hello", "!"]
3. **Number handling**: "123" → ["123"] (keep as one token)
4. **Case handling**: Keep or lowercase? (BERT lowercases, GPT doesn't)

**Example (GPT-2 pre-tokenization)**:
```
Input: "Hello, world! 123"
Pre-tokenize: ["Hello", ",", " world", "!", " 123"]
Note: Spaces are preserved as part of tokens
```

**SentencePiece pre-tokenization**:
```
Input: "Hello world"
Pre-tokenize: ["▁Hello", "▁world"]
Note: Leading space encoded as ▁ (U+2581)
```

**Why pre-tokenization matters**:
- Ensures consistent splitting
- Handles edge cases (URLs, emails, hashtags)
- Language-specific rules (Chinese: no spaces, need segmentation)

---

## 4. Special Tokens

### 4.1 Padding Token (PAD)

**Purpose**: Make sequences equal length (for batching)

**Example**:
```
Sequence 1: [15, 23, 45, 67]       Length: 4
Sequence 2: [12, 34]               Length: 2

Padded (max length = 4):
Sequence 1: [15, 23, 45, 67]
Sequence 2: [12, 34, 0, 0]         Padded with <PAD> (ID 0)
```

**Critical detail**: Padding is ignored in attention
```python
# Attention mask
Mask 1: [1, 1, 1, 1]  # Attend to all tokens
Mask 2: [1, 1, 0, 0]  # Ignore padding (0s)
```

**Best practices**:
- Pad to right (for generation tasks)
- Pad to left (for encoder models like BERT)

### 4.2 End-of-Sequence (EOS / SEP)

**Purpose**: Signal end of text or separation between segments

**Names vary by model**:
- GPT: `<|endoftext|>` (ID 50256 in GPT-2)
- BERT: `[SEP]` (ID 102)
- Llama: `</s>` (ID 2)

**Usage in generation**:
```
Prompt: "The capital of France is"
Generated: "The capital of France is Paris<EOS>"
           ↑ Stop generation when <EOS> is emitted
```

**Usage in BERT (sentence pair)**:
```
Input: "[CLS] Question? [SEP] Answer. [SEP]"
Purpose: Separate question from answer
```

### 4.3 Beginning-of-Sequence (BOS / CLS)

**Purpose**: Mark start of sequence

**GPT models**: No BOS token (start directly with text)
**BERT**: `[CLS]` token (ID 101)
- Used for classification (embedding of [CLS] = sentence embedding)

**Llama**: `<s>` (ID 1)

**Example (BERT)**:
```
Input: "Hello world"
Tokenized: "[CLS] Hello world [SEP]"
IDs: [101, 7592, 2088, 102]
```

### 4.4 Unknown Token (UNK)

**Purpose**: Handle out-of-vocabulary words

**When used**:
- Word-level tokenizers (can't represent unseen words)
- Character-level: Never needed
- BPE: Rare (only for bytes outside training data)

**Example (Word-level)**:
```
Vocabulary: ["the", "cat", "sat"]
Input: "the dog sat"
Tokenized: ["the", "<UNK>", "sat"]
Problem: Lost information about "dog"
```

**Byte-level BPE avoids UNK**:
- Any text can be encoded as bytes
- Fallback: Represent as individual bytes

### 4.5 Mask Token (MASK)

**Purpose**: Masked language modeling (BERT pretraining)

**Example**:
```
Input: "The cat [MASK] on the mat"
Task: Predict [MASK] → "sat"
```

**Used by**: BERT, RoBERTa, ALBERT, ELECTRA

**Not used in GPT** (autoregressive, no masking)

### 4.6 Special Token Handling in Generation

**Key challenge**: Prevent model from generating special tokens inappropriately

**Bad**:
```
Prompt: "Hello"
Generated: "Hello <PAD> <PAD> world"  # Should never happen
```

**Solution**: Mask special tokens during sampling
```python
# Logits for next token
logits = model(input_ids)

# Mask special tokens (PAD, BOS, EOS)
logits[:, [0, 1, 2]] = -float('inf')  # Never sample these

# Sample next token
next_token = torch.argmax(logits, dim=-1)
```

**Exception**: Allow EOS when stopping generation
```python
if max_length_reached or user_stopped:
    allow_eos = True
```

### 4.7 Token IDs and Vocabularies

**Token → ID mapping**:
```
Vocabulary:
0: <PAD>
1: <BOS>
2: <EOS>
3: <UNK>
4: "the"
5: "cat"
...
50256: "Hello"
```

**Encoding**:
```
"Hello world" → [50256, 995]
```

**Decoding**:
```
[50256, 995] → "Hello world"
```

**Critical detail**: IDs are arbitrary
- Different models use different IDs for same token
- Can't share embeddings between models with different vocabularies

---

## 5. Tokenization Challenges

### 5.1 Out-of-Vocabulary (OOV) Handling

**Problem**: Unseen words/characters

**Solutions by tokenizer type**:

**Word-level**: Use `<UNK>` (loses information)

**BPE/WordPiece**: Decompose into subwords
```
"unbelievable" (unseen)
→ ["un", "believ", "able"] (all in vocabulary)
```

**Byte-level BPE**: Always works (fallback to bytes)
```
"🤖" (emoji, unseen)
→ Encoded as UTF-8 bytes: [240, 159, 164, 150]
→ Each byte has a token
```

**Real-world example (GPT-4)**:
- Trained on English-heavy corpus
- User inputs Chinese text
- Byte-level BPE: Works (each Chinese character is ~3 bytes)
- Result: 3x more tokens than English, but functional

### 5.2 Multilingual Tokenization

**Challenge**: Different languages have different structures

**English**: Space-separated words
**Chinese**: No spaces between words (need word segmentation)
**German**: Compound words (`Donaudampfschifffahrtsgesellschaft`)
**Arabic**: Right-to-left, diacritics

**Solutions**:

**SentencePiece**: Language-agnostic (treats all text as byte stream)

**Multilingual BPE** (mBERT, XLM-R):
- Train on corpus from 100+ languages
- Vocabulary covers common subwords across languages
- Example: "hello" (English), "hola" (Spanish) share "hel", "ol"

**Example (XLM-RoBERTa)**:
```
Vocabulary: 250K tokens
Coverage: 100 languages
Result: 2-3x more tokens for non-English text
```

### 5.3 Language-Specific Considerations

**Chinese/Japanese/Korean**:
- No whitespace between words
- Pre-tokenization: Use segmenter (jieba for Chinese)
- Or: Treat each character as potential token boundary (SentencePiece)

**Arabic/Hebrew**:
- Right-to-left (RTL) text
- Diacritics (optional accent marks)
- Solution: Normalize (remove diacritics) or preserve (depends on task)

**German**:
- Long compound words (`Rindfleischetikettierungsüberwachungsaufgabenübertragungsgesetz`)
- BPE handles well (breaks into subwords)

**Turkish**:
- Agglutinative language (words formed by suffixes)
- Example: `evlerimizden` = "from our houses"
- Unigram LM works better than BPE (probabilistic segmentation)

### 5.4 Domain-Specific Vocabulary

**Problem**: General tokenizers inefficient for specialized domains

**Code**:
- Need tokens for `==`, `!=`, `++`, `<=`
- Camel case: `getUserName` → `["get", "User", "Name"]`
- Solution: Train tokenizer on code corpus (Codex, CodeGen)

**Biomedical**:
- Long technical terms: `hydroxychloroquine`, `COVID-19`
- Acronyms: `SARS-CoV-2`, `mRNA`
- Solution: BioBERT (trained on PubMed)

**Legal**:
- Formal language: `hereinafter`, `notwithstanding`
- Latin terms: `pro bono`, `habeas corpus`
- Solution: LegalBERT

**Example (Codex vs GPT-3 tokenization)**:
```
Code: "def get_user_name():"

GPT-3: ["def", " get", "_user", "_name", "():"]  # 5 tokens
Codex: ["def", " get_user_name", "():"]          # 3 tokens

Result: 40% fewer tokens for code with Codex
```

### 5.5 Rare Word Handling

**Long-tail problem**: 80% of tokens cover 20% of vocabulary

**Strategies**:

**1. Subword decomposition** (BPE default):
```
"antidisestablishmentarianism" (rare)
→ ["anti", "dis", "establish", "ment", "arian", "ism"]
```

**2. Character fallback** (SentencePiece with character coverage):
```
Rare Unicode character: ⚛ (atom symbol)
→ Decompose to bytes if not in vocabulary
```

**3. Frequency threshold**:
```
Min frequency = 2 during training
Tokens appearing once: Likely noise, exclude from vocabulary
```

**Trade-off**:
- More aggressive subword splitting: Longer sequences, but covers rare words
- Less splitting: Shorter sequences, but more UNK tokens

### 5.6 Subword vs Word-Level Trade-offs

**Word-level**:
- ✅ Shortest sequences (1 word = 1 token)
- ✅ Clear semantic boundaries
- ❌ Huge vocabulary (millions of words)
- ❌ OOV problem
- ❌ Can't handle morphology

**Subword (BPE/WordPiece)**:
- ✅ Balanced vocabulary (30K-100K)
- ✅ No OOV (can represent any word)
- ✅ Handles morphology (`play` + `ing`)
- ❌ Longer sequences (1 word = 1-3 tokens)
- ❌ Ambiguity (multiple valid segmentations)

**Character-level**:
- ✅ Tiny vocabulary (~100 characters)
- ✅ No OOV, handles typos
- ❌ 5-10x longer sequences
- ❌ Harder to learn semantics

**When to use which**:
- Word-level: Small-scale, controlled vocabulary (e.g., chatbot with fixed domain)
- Subword: General-purpose LLMs (default choice)
- Character: Low-resource languages, noisy text (OCR, social media)

### 5.7 Tokenization Consistency

**Critical problem**: Train/inference mismatch

**Example**:
```
Training tokenizer on: "Hello world"
Learned merges: ("Hello", " world") → "Hello world"

Inference on: "Hello  world" (two spaces)
Tokenization: ["Hello", "  world"] (different!)
Result: Model sees unseen token pattern
```

**Solutions**:

**1. Normalize whitespace**:
```python
text = " ".join(text.split())  # Replace multiple spaces with single
```

**2. Preserve pre-tokenization**:
```python
# Save pre-tokenization rules with tokenizer
tokenizer.save_pretrained("my_model/")
# Load and apply same rules at inference
tokenizer = AutoTokenizer.from_pretrained("my_model/")
```

**3. Byte-level BPE**:
- Treats all text as bytes
- No normalization needed
- Consistent encoding

**Real-world bug (GPT-2)**:
- Training: Lowercased text
- Inference: Kept original case
- Result: Poor performance on proper nouns
- Fix: Always keep case (GPT-3+)

---

## 6. Tokenization Impact

### 6.1 Vocabulary Size vs Computation Trade-off

**Key insight**: Vocabulary size directly affects model size and speed

**Embedding matrix size**:
```
Vocab size × Embedding dim × 4 bytes (FP32)

GPT-2 (50K vocab, 768 dim):
50,000 × 768 × 4 = 153 MB

GPT-4 (~100K vocab, 12,288 dim):
100,000 × 12,288 × 4 = 4.9 GB (!!)
```

**Output layer size** (same as embedding):
```
Total embedding + output = 2 × Vocab × Embed dim
GPT-4: 2 × 4.9 GB = 9.8 GB just for token embeddings
```

**The 100K vocabulary problem**:

**Why larger vocabulary seems better**:
- Shorter sequences (fewer tokens per sentence)
- Faster inference (fewer autoregressive steps)
- Better compression (1 token represents more text)

**Why it's actually worse**:
- **Memory**: Embedding matrix explodes
- **Speed**: Softmax over 100K classes is slow
  ```
  Softmax computation: O(vocab_size)
  100K vocab: 2x slower than 50K
  ```
- **Training**: More parameters to learn
- **Sampling**: Harder to sample from large distribution

**Real-world example (Llama 3)**:
- Llama 2: 32K vocabulary
- Llama 3: 128K vocabulary (4x larger)
- Why? Better multilingual support (100+ languages)
- Cost: 4x larger embedding matrix
- Trade-off: Shorter sequences for non-English text

**Optimal vocabulary size** (from research):
- English-only: 30K-50K (GPT-2, GPT-3)
- Multilingual: 100K-250K (XLM-R, mT5)
- Code: 50K-100K (Codex)

**Rule of thumb** (from Karpathy):
> "Don't blindly increase vocabulary. Each doubling of vocab size adds ~500 MB to model size (for 12B param model) and slows inference by ~20%."

### 6.2 Performance Impact on Model

**Sequence length affects**:

**1. Attention complexity**: O(n²) where n = sequence length
```
50 tokens: 2,500 attention computations
100 tokens: 10,000 attention computations (4x)
```

**2. Memory usage**: KV cache grows with sequence length
```
KV cache size = 2 × layers × heads × head_dim × seq_len × batch

For 7B model at 2K context:
2 × 32 × 32 × 128 × 2048 × 1 × 2 bytes = 1 GB

For 4K context: 2 GB (doubles!)
```

**3. Training time**: Longer sequences = slower training
```
GPT-3 training:
50-token sequences: 1 day
100-token sequences: 2+ days (quadratic scaling)
```

**Vocabulary size affects**:

**1. Embedding lookup**: Negligible (hash table lookup = O(1))

**2. Softmax computation**: O(vocab_size)
```
50K vocab: 50K exponentials + sum
100K vocab: 100K exponentials + sum (2x slower)
```

**3. Model size**: Linear with vocabulary
```
1K vocab increase = ~50 MB model size (for 7B param model)
```

### 6.3 Inference Speed Considerations

**Bottlenecks**:

**1. Autoregressive generation**: Generate one token at a time
```
Sequence of 100 tokens = 100 forward passes
Vocabulary size = 50K: 50K softmax per pass
Total: 5 million softmax computations
```

**2. Memory bandwidth**: Moving KV cache to/from GPU memory
```
7B model at 2K context:
1 GB KV cache per forward pass
At 900 GB/s bandwidth: ~1ms per token (memory-bound)
```

**3. Batch size**: Larger vocabulary = smaller batch size (GPU memory limit)
```
Available memory: 24 GB (A100)
Embedding matrix: 5 GB (100K vocab)
Remaining: 19 GB for batch processing
Batch size: 8 (vs 16 with 50K vocab)
```

**Optimization strategies**:

**1. Vocabulary pruning**: Remove rare tokens
```
Original: 100K tokens
Pruned: 50K most frequent (covers 99% of text)
Result: 2x faster inference
```

**2. Adaptive softmax**: Different computation for frequent vs rare tokens
```
Frequent 10K tokens: Full softmax
Rare 90K tokens: Approximate (hierarchical softmax)
Speedup: 3-5x
```

**3. Quantization**: FP16 or INT8 embeddings
```
FP32: 4 bytes per parameter
FP16: 2 bytes (2x memory reduction)
INT8: 1 byte (4x reduction)
```

### 6.4 Embedding Matrix Memory Footprint

**Calculation**:
```
Memory = Vocab size × Embedding dim × Bytes per parameter

GPT-2 (FP32):
50,257 × 768 × 4 = 154 MB

GPT-3 175B (FP32):
50,257 × 12,288 × 4 = 2.4 GB

Llama 3 8B (BF16):
128,256 × 4,096 × 2 = 1.05 GB
```

**Impact on model size**:
```
Total model size = Embedding + Attention + FFN

Llama 3 8B:
Embeddings: 1.05 GB (6.5%)
Attention: 8 GB (50%)
FFN: 7 GB (43.5%)
Total: 16 GB
```

**Why embeddings matter less for large models**:
- 7B model: Embeddings = 10% of model size
- 175B model: Embeddings = 1% of model size
- Conclusion: Vocabulary size matters more for small models

**Memory optimization**:
```
# Tie input and output embeddings (share weights)
Input embeddings: 1 GB
Output embeddings: 1 GB (same weights)
Tied: 1 GB (save 1 GB)

Used by: GPT-2, BERT, T5
```

### 6.5 Compression Ratio

**Definition**: How many characters per token (on average)

**Calculation**:
```
Compression ratio = Total characters / Total tokens

Example:
Text: "Hello, world!" (13 characters)
Tokens: ["Hello", ",", " world", "!"] (4 tokens)
Ratio: 13 / 4 = 3.25 chars/token
```

**Typical ratios**:
- English (GPT-2): 4-5 chars/token
- Code (Codex): 3-4 chars/token
- Chinese (GPT-2): 1.5-2 chars/token (worse!)
- Chinese (Llama 3): 2.5-3 chars/token (better)

**Why compression matters**:
```
Context window: 4096 tokens

English: 4096 × 4 = 16,384 characters (~3 pages)
Chinese (GPT-2): 4096 × 1.5 = 6,144 characters (~1 page)
Chinese (Llama 3): 4096 × 2.5 = 10,240 characters (~2 pages)
```

**Improving compression**:
- Train on target language (more tokens for that language)
- Larger vocabulary (more specific tokens)
- Better pre-tokenization (handle language-specific rules)

### 6.6 Token-to-Character Ratio

**English (GPT-4)**:
```
"The quick brown fox jumps" (25 chars)
→ ["The", " quick", " brown", " fox", " jumps"] (5 tokens)
Ratio: 25/5 = 5 chars/token
```

**Code (Codex)**:
```
"def hello():" (12 chars)
→ ["def", " hello", "(", ")", ":"] (5 tokens)
Ratio: 12/5 = 2.4 chars/token
```

**Chinese (GPT-2)**:
```
"你好世界" (4 chars)
→ ["你", "好", "世", "界"] (4 tokens, worst case: each char is 3 byte-level BPE tokens)
→ Actually: 12 tokens (3 per character)
Ratio: 4/12 = 0.33 chars/token (terrible!)
```

**Chinese (Llama 3, with better Chinese support)**:
```
"你好世界" (4 chars)
→ ["你好", "世界"] (2 tokens)
Ratio: 4/2 = 2 chars/token (much better!)
```

**Why this matters for pricing**:
```
GPT-4 API: $0.03 per 1K tokens

English text (5 chars/token):
1000 tokens = 5000 characters = ~1 page
Cost: $0.03 per page

Chinese text (0.33 chars/token with GPT-2):
1000 tokens = 330 characters = ~0.1 page
Cost: $0.03 per 0.1 page = $0.30 per page (10x more expensive!)
```

**Solution**: Use models with better multilingual support (Llama 3, GPT-4 with larger vocab)

---

## 7. Production Case Studies

### 7.1 GPT-2 to GPT-3 Evolution

**GPT-2** (2019):
- Vocabulary: 50,257 tokens
- Tokenizer: Byte-level BPE
- Training corpus: WebText (40 GB, Reddit links)
- Compression: ~4 chars/token (English)

**GPT-3** (2020):
- Vocabulary: 50,257 tokens (same as GPT-2!)
- Tokenizer: Byte-level BPE (same)
- Training corpus: 570 GB (45% CommonCrawl, 19% WebText2, etc.)
- Why no change? Prioritized model scaling over tokenizer

**Lesson**: Vocabulary isn't always the bottleneck. GPT-3's 175B parameters mattered more than tokenizer improvements.

### 7.2 Llama 2 to Llama 3 Evolution

**Llama 2** (2023):
- Vocabulary: 32,000 tokens
- Tokenizer: SentencePiece BPE
- Focus: English + some multilingual
- Chinese compression: ~1.5 chars/token

**Llama 3** (2024):
- Vocabulary: 128,256 tokens (4x larger!)
- Tokenizer: Improved SentencePiece BPE
- Training corpus: 15 trillion tokens (100+ languages)
- Chinese compression: ~2.5 chars/token (67% improvement!)

**Why the change?**:
- Meta wanted better multilingual support
- Trade-off: 4x larger embedding matrix (cost: +1 GB model size)
- Benefit: 30-50% fewer tokens for non-English text

**Performance impact**:
```
Llama 2 on Chinese text:
Inference speed: 20 tokens/sec
Context limit: 4K tokens = ~6K characters

Llama 3 on Chinese text:
Inference speed: 20 tokens/sec (same)
Context limit: 8K tokens = ~20K characters (3x more text!)
```

### 7.3 GPT-4 Tokenization Strategy

**Challenges**:
- Multimodal (text + images)
- 100+ languages
- Code + natural language
- Long context (128K tokens)

**Solution**:
- Vocabulary: ~100K tokens (estimated, not public)
- Tokenizer: tiktoken (byte-level BPE, optimized in Rust)
- Special handling:
  - Code tokens: `++`, `==`, `<=` as single tokens
  - Multilingual: More tokens for non-English
  - Whitespace: Preserved (important for code indentation)

**Performance**:
```
English: 4-5 chars/token
Code: 3-4 chars/token
Chinese: 2-3 chars/token (better than GPT-3)

Cost calculation (API):
$0.03 per 1K tokens (8K context)
$0.06 per 1K tokens (32K context)

For 10-page document (50K characters):
Tokens: 50K / 4 = 12.5K tokens
Cost: $0.03 × 12.5 = $0.375
```

### 7.4 Claude (Anthropic) Tokenization

**Approach**: Similar to GPT-4
- Vocabulary: ~100K tokens
- Tokenizer: Custom BPE (details not public)
- Long context: 200K tokens (vs GPT-4's 128K)

**Key difference**: Focus on safety
- Special tokens for constitutional AI
- Filtered vocabulary (no offensive tokens)
- Careful handling of code (prevent prompt injection)

**Performance**:
```
200K context window:
= 800K characters (English, at 4 chars/token)
= ~400 pages of text
= 2-3 novels
```

**Use case**: Long document analysis (contracts, research papers, codebases)

### 7.5 Code-Specific Tokenizers (Codex, StarCoder)

**Challenge**: Code has different statistics than natural language

**Codex (OpenAI)**:
- Modified GPT-3 tokenizer
- Additional tokens for code syntax:
  - Operators: `++`, `--`, `==`, `!=`, `<=`, `>=`
  - Brackets: `()`, `[]`, `{}`
  - Indentation: 4 spaces = single token
- Camel case splitting: `getUserName` → `["get", "User", "Name"]`

**StarCoder (BigCode)**:
- Vocabulary: 49,152 tokens
- Training: GitHub code (80+ languages)
- Tokenizer: SentencePiece BPE
- Result: 30-40% fewer tokens than GPT-3 on code

**Example**:
```python
Code: "def get_user_name():"

GPT-3: ["def", " get", "_user", "_name", "():"]  # 5 tokens
Codex: ["def", " get_user_name", "():"]          # 3 tokens
StarCoder: ["def", " get_user_name", "():"]      # 3 tokens

Efficiency gain: 40% fewer tokens = 40% faster inference
```

### 7.6 Multilingual Models (mT5, XLM-R)

**mT5 (Google)**:
- Vocabulary: 250,000 tokens (huge!)
- Tokenizer: SentencePiece (Unigram LM)
- Coverage: 101 languages
- Why so large? Each language needs ~2K tokens, 100 langs = 200K+

**XLM-RoBERTa (Meta)**:
- Vocabulary: 250,000 tokens
- Tokenizer: SentencePiece BPE
- Training: CommonCrawl (2.5TB, 100 languages)
- Result: Near-native performance on all 100 languages

**Trade-offs**:
```
Vocabulary size: 250K
Embedding matrix: 250K × 768 × 4 = 750 MB
Output layer: 750 MB
Total: 1.5 GB just for embeddings (vs 300 MB for GPT-2)

Cost: 5x larger embedding matrices
Benefit: Equal performance across languages
```

### 7.7 Domain Adaptation: BioBERT, LegalBERT

**BioBERT** (Medical domain):
- Start: BERT vocabulary (30K tokens, general English)
- Problem: Medical terms split poorly
  - `hydroxychloroquine` → 4-5 tokens
  - `COVID-19` → 3 tokens
- Solution: Continue training BERT tokenizer on PubMed corpus
- New vocabulary: 30K tokens + 5K medical terms = 35K
- Result: Medical terms are 1-2 tokens

**LegalBERT** (Legal domain):
- Start: BERT vocabulary
- Problem: Legal jargon split poorly
  - `notwithstanding` → 3 tokens
  - `hereinafter` → 3 tokens
- Solution: Train tokenizer on legal corpus (court opinions, contracts)
- Result: 20% fewer tokens on legal text

**Lesson**: Domain-specific tokenizers improve efficiency by 20-40%

---

## 8. Interview Questions

### 8.1 Fundamental Concepts

**Q1: What is tokenization and why do we need it?**

**Expected answer**:
Tokenization converts text into discrete units (tokens) that can be represented as integers for neural network processing. We need it because:
1. Neural networks operate on numbers, not text
2. Character-level is too long (5-10x more tokens)
3. Word-level has vocabulary explosion and OOV problems
4. Subword tokenization (BPE) balances vocabulary size and sequence length

**Q2: Explain how BPE (Byte Pair Encoding) works.**

**Expected answer**:
BPE is an iterative algorithm:
1. Initialize vocabulary with characters (or bytes)
2. Find the most frequent adjacent pair in the corpus
3. Merge this pair into a new token, add to vocabulary
4. Repeat until desired vocabulary size

Example: "low low low" → Learn merge ("l", "o") → "lo w lo w lo w" → Learn merge ("lo", "w") → "low low low"

At inference, apply merges in order learned during training.

**Q3: What's the difference between BPE and WordPiece?**

**Expected answer**:
- **BPE**: Frequency-based (merge most frequent pair)
- **WordPiece**: Likelihood-based (merge pair that maximizes corpus likelihood)
- BPE is simpler and faster to train
- WordPiece is more principled (considers probability)
- Both produce similar results in practice

**Q4: Why does GPT use byte-level BPE?**

**Expected answer**:
Byte-level BPE operates on UTF-8 bytes (0-255) instead of characters:
- Universal: Can tokenize any language, any character
- No UNK tokens: Every text can be encoded as bytes
- No preprocessing needed (no lowercasing, no accent stripping)
- Robust to noise, typos, emoji, code

Trade-off: Slightly longer sequences for non-ASCII text

### 8.2 Tokenization Methods

**Q5: Compare SentencePiece to standard BPE.**

**Expected answer**:
SentencePiece is language-agnostic:
- Treats text as raw byte stream (no pre-tokenization)
- Encodes whitespace as special character `▁` (reversible)
- Works for languages with no spaces (Chinese, Japanese, Thai)
- Supports both BPE and Unigram LM algorithms

Standard BPE requires language-specific pre-tokenization (split on whitespace).

**Q6: When would you use character-level tokenization?**

**Expected answer**:
Character-level is rare in modern LLMs, but useful for:
- Very low-resource languages (no training data for BPE)
- Noisy text (OCR errors, social media typos)
- When vocabulary size must be minimal (<100 tokens)

Trade-off: 5-10x longer sequences, harder to learn semantics

**Q7: How does tokenization differ for code vs natural language?**

**Expected answer**:
Code tokenization needs:
- Syntax awareness: `(`, `)`, `{`, `}`, `;` should be separate tokens
- Camel case splitting: `getUserName` → `["get", "User", "Name"]`
- Indentation preservation: 4 spaces = single token (for Python)
- Operator handling: `++`, `==`, `<=` as single tokens

Natural language: Whitespace and punctuation are primary boundaries

### 8.3 Special Tokens and Vocabulary

**Q8: What are special tokens and why are they needed?**

**Expected answer**:
Special tokens mark boundaries and metadata:
- `<PAD>`: Padding for batching (ignored in attention)
- `<BOS>`/`<CLS>`: Beginning of sequence
- `<EOS>`/`<SEP>`: End of sequence or separator
- `<UNK>`: Unknown token (word-level tokenizers)
- `<MASK>`: Masked token (BERT pretraining)

They're needed for:
- Variable-length sequences (padding)
- Multi-segment inputs (BERT's [CLS] question [SEP] answer [SEP])
- Stopping generation (model outputs <EOS>)

**Q9: How does padding work and why does it matter?**

**Expected answer**:
Padding makes sequences equal length for batching:
```
Seq 1: [15, 23, 45] → [15, 23, 45, 0, 0]
Seq 2: [12, 34, 56, 78, 89] → [12, 34, 56, 78, 89]
```

Padding tokens are masked out in attention (attention mask = [1, 1, 1, 0, 0]).

Matters because:
- Allows efficient batch processing on GPU
- Must be ignored in loss computation (otherwise model learns to predict padding)

**Q10: Why might a model have a vocabulary of 100K tokens instead of 50K?**

**Expected answer**:
Larger vocabulary:
- **Pros**: Shorter sequences (fewer tokens per sentence), better compression, faster generation
- **Cons**: Larger embedding matrix (2x memory for 2x vocab), slower softmax (2x compute), more parameters to learn

100K vocab is justified for:
- Multilingual models (need tokens for 100+ languages)
- Long-context models (shorter sequences fit more text in context window)
- Domain-specific models (medical, legal terms as single tokens)

### 8.4 Tokenization Challenges

**Q11: What is the "out-of-vocabulary" problem and how do modern tokenizers solve it?**

**Expected answer**:
OOV occurs when a word wasn't seen during tokenizer training.

- **Word-level**: Replace with `<UNK>` (loses information)
- **BPE/WordPiece**: Decompose into subwords (always works)
  - Example: `"unbelievable"` → `["un", "believ", "able"]`
- **Byte-level BPE**: Fallback to bytes (never fails)
  - Example: Emoji `🤖` → UTF-8 bytes → tokens

Modern LLMs (GPT-2+) use byte-level BPE, so OOV is impossible.

**Q12: How does tokenization affect multilingual models?**

**Expected answer**:
Challenges:
- English-centric tokenizers inefficient for non-English (more tokens per sentence)
- Different languages need different pre-tokenization (Chinese: no spaces)

Solutions:
- Train on multilingual corpus (XLM-R, mT5)
- Use SentencePiece (language-agnostic, no pre-tokenization)
- Larger vocabulary (250K tokens to cover 100+ languages)

Result: 2-3x more tokens for non-English text, but functional

**Q13: What is training-serving skew in tokenization?**

**Expected answer**:
Mismatch between tokenization during training vs inference.

Example:
- Training: Lowercase all text, then tokenize
- Inference: Forget to lowercase, tokenize raw text
- Result: Model sees different token distributions, poor performance

Prevention:
- Save tokenizer configuration with model
- Apply exact same pre-processing at inference
- Use byte-level BPE (minimal pre-processing, more robust)

### 8.5 Performance and Trade-offs

**Q14: How does vocabulary size affect model performance and efficiency?**

**Expected answer**:

**Memory**:
- Embedding matrix size = `vocab_size × embed_dim × 4 bytes`
- 50K vocab, 768 dim: 153 MB
- 100K vocab, 768 dim: 306 MB (2x memory)

**Speed**:
- Softmax complexity: O(vocab_size)
- 100K vocab: 2x slower than 50K in output layer

**Sequence length**:
- Larger vocab: Fewer tokens per sentence
- Attention complexity: O(seq_len²)
- Trade-off: Larger vocab = shorter sequences (faster attention) but slower softmax

**Optimal**: 30K-50K for English, 100K-250K for multilingual

**Q15: Calculate the memory footprint of a tokenizer's embedding matrix.**

**Expected answer**:
Formula: `Vocab size × Embedding dimension × Bytes per parameter`

Example (GPT-2):
- Vocabulary: 50,257 tokens
- Embedding dimension: 768
- Precision: FP32 (4 bytes)
- Memory: 50,257 × 768 × 4 = 154 MB

For FP16: Divide by 2 → 77 MB
For INT8: Divide by 4 → 38.5 MB

Note: This is just embeddings. Total model includes attention, FFN layers.

**Q16: Why does Chinese text use 2-3x more tokens than English in GPT-2?**

**Expected answer**:
GPT-2 uses byte-level BPE trained on English-heavy corpus:
- English words: Frequent, learned as single tokens (`"the"`, `"is"`)
- Chinese characters: Rare in training, decomposed to UTF-8 bytes
- Each Chinese character = 3 bytes in UTF-8
- Result: 1 character = 3 tokens (worst case)

Llama 3 fixes this:
- Trained on multilingual corpus (100+ languages)
- Vocabulary: 128K tokens (includes Chinese characters as single tokens)
- Result: 1 character ≈ 1 token (same as English)

### 8.6 Practical Implementation

**Q17: Walk me through training a tokenizer from scratch.**

**Expected answer**:
1. **Corpus selection**: Collect representative text (10GB+ recommended)
2. **Pre-tokenization**: Split on whitespace and punctuation
3. **Initialize**: Start with character or byte vocabulary (256 for byte-level)
4. **Iterative merging**: Apply BPE algorithm
   - Count all adjacent pairs
   - Merge most frequent pair
   - Repeat until vocab_size reached
5. **Save**: Store vocabulary and merge rules

Code (HuggingFace Tokenizers):
```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

tokenizer = Tokenizer(BPE())
trainer = BpeTrainer(vocab_size=30000, special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"])
tokenizer.train(files=["corpus.txt"], trainer=trainer)
tokenizer.save("my_tokenizer.json")
```

**Q18: How would you debug a model that performs poorly on a specific domain?**

**Expected answer**:
Check tokenization first:
1. **Tokenize sample text**: See how domain-specific terms are split
   - Medical: `"hydroxychloroquine"` → Too many tokens?
   - Legal: `"notwithstanding"` → Inefficient splitting?
2. **Compare compression ratios**: Domain text vs training corpus
   - If 2x more tokens, tokenizer is inefficient
3. **Solutions**:
   - **Option 1**: Train domain-specific tokenizer (BioBERT approach)
   - **Option 2**: Continue training existing tokenizer on domain corpus
   - **Option 3**: Fine-tune model (accept tokenization inefficiency)

**Q19: How do you handle long documents that exceed the context window?**

**Expected answer**:
Tokenization makes this worse (documents become longer in tokens):
- 10,000 words at 1.5 words/token = 15,000 tokens
- If context window = 4,096 tokens, need to truncate

Solutions:
1. **Chunking**: Split document into overlapping chunks
   - Chunk 1: Tokens 0-4096
   - Chunk 2: Tokens 2048-6144 (50% overlap)
   - Process each chunk separately, aggregate results
2. **Sliding window**: Move window across document
3. **Summarization**: Summarize first, then process summary
4. **Sparse attention**: Longformer, BigBird (attend to subset of tokens)

### 8.7 Advanced Topics

**Q20: Explain the concept of "compression ratio" in tokenization.**

**Expected answer**:
Compression ratio = Average characters per token

**English (GPT-2)**: 4-5 chars/token
- "Hello world" (11 chars) → ["Hello", " world"] (2 tokens) = 5.5 chars/token

**Code (Codex)**: 3-4 chars/token
- Code has more symbols, operators (each is a token)

**Chinese (GPT-2)**: 1.5-2 chars/token
- Poor compression due to English-centric training

**Why it matters**:
- Context window: 4,096 tokens at 4 chars/token = 16,384 characters
- Chinese at 1.5 chars/token = 6,144 characters (much less!)

Better compression = more text fits in context window

**Q21: How does tokenization interact with the model's embedding layer?**

**Expected answer**:
Tokenization → Token IDs → Embedding lookup

```
Text: "Hello world"
Tokenize: ["Hello", " world"]
Token IDs: [15496, 995]
Embedding lookup: [E_15496, E_995] (each is a d-dimensional vector)
```

Embedding matrix: `[vocab_size, embedding_dim]`
- GPT-2: [50257, 768]
- Each token ID maps to a learned vector

Training:
- Embeddings are learned jointly with the model
- Frequent tokens get more gradient updates (more refined embeddings)
- Rare tokens: Less training signal, noisier embeddings

**Q22: What is the relationship between vocabulary size and model generalization?**

**Expected answer**:

**Small vocabulary** (10K-30K):
- Pros: Fewer parameters, faster training, better for low-resource languages
- Cons: Longer sequences (more aggressive subword splitting), less efficient

**Large vocabulary** (100K+):
- Pros: Shorter sequences, better compression, more specific tokens
- Cons: More parameters to learn, need more data, risk of overfitting rare tokens

**Trade-off**:
- Frequent tokens (top 10K): Always beneficial to include
- Rare tokens (bottom 50%): May not learn good embeddings (not enough data)
- Optimal: Depends on training data size and diversity

**Rule of thumb**: Vocabulary size should scale with training corpus size
- 10GB corpus: 30K-50K vocab
- 100GB corpus: 50K-100K vocab
- 1TB+ corpus: 100K-250K vocab (multilingual models)

### 8.8 Production and Debugging

**Q23: Your model is generating gibberish. How do you debug if tokenization is the issue?**

**Expected answer**:
Check these:
1. **Token IDs in range**: Ensure IDs are within [0, vocab_size)
   - Out-of-range IDs cause embedding lookup errors
2. **Decode tokens**: Convert token IDs back to text
   - If decoding fails, tokenizer is corrupted
3. **Compare train vs inference**: Same tokenizer?
   - Different tokenizers = different token IDs = garbage
4. **Special tokens**: Are they handled correctly?
   - Model should NOT generate `<PAD>` or `<BOS>` mid-sequence
5. **Vocabulary mismatch**: Did you load correct tokenizer?
   - GPT-2 tokenizer on Llama model = disaster

**Q24: How would you optimize inference speed for a tokenization-heavy pipeline?**

**Expected answer**:
Tokenization is usually not the bottleneck (model inference is), but for high-throughput systems:

1. **Batch tokenization**: Tokenize multiple texts at once
   ```python
   # Slow: Loop
   for text in texts:
       tokens = tokenizer.encode(text)
   
   # Fast: Batch
   tokens = tokenizer.batch_encode(texts)
   ```

2. **Use fast tokenizers**: HuggingFace Fast Tokenizers (Rust-based)
   - 5-10x faster than Python tokenizers

3. **Cache tokenization**: If same prompts repeated
   ```python
   cache = {}
   if text in cache:
       tokens = cache[text]
   else:
       tokens = tokenizer.encode(text)
       cache[text] = tokens
   ```

4. **Pre-tokenize offline**: For static datasets
   - Tokenize once, save token IDs
   - At training/inference, load pre-tokenized data

5. **Parallel tokenization**: Multi-threading or multi-processing

**Q25: Explain the trade-off between vocabulary size and sequence length in the context of attention complexity.**

**Expected answer**:
Attention complexity: O(n² × d) where n = sequence length, d = model dimension

**Small vocabulary** (30K):
- Longer sequences (more aggressive subword splitting)
- "unbelievable" → ["un", "be", "liev", "able"] (4 tokens)
- 100-word paragraph = 150 tokens
- Attention: O(150² × d) = O(22,500 × d)

**Large vocabulary** (100K):
- Shorter sequences (fewer tokens per word)
- "unbelievable" → ["unbelievable"] (1 token)
- 100-word paragraph = 100 tokens
- Attention: O(100² × d) = O(10,000 × d)

**Trade-off**:
- Large vocab: Faster attention (shorter sequences), slower softmax (more classes)
- Small vocab: Slower attention (longer sequences), faster softmax (fewer classes)

**Optimal**:
- For long documents (>1000 words): Large vocabulary wins (attention is bottleneck)
- For short documents (<100 words): Small vocabulary may be better (softmax is cheap)

---

## Summary: Key Takeaways for Interviews

### Must-Know Concepts:
1. **BPE algorithm**: Frequency-based merging
2. **Byte-level BPE**: Universal, no UNK tokens
3. **Special tokens**: PAD, BOS, EOS, UNK, MASK
4. **Vocabulary size trade-offs**: Memory, speed, sequence length
5. **Compression ratio**: Chars per token (4-5 for English)
6. **Multilingual challenges**: SentencePiece, larger vocabularies
7. **Training-serving consistency**: Use same tokenizer!

### Common Traps:
1. **Vocabulary size ≠ Always better**: Larger vocab = more memory, slower softmax
2. **Byte-level BPE ≠ Character-level**: Bytes (256) vs characters (100K+)
3. **Tokenization is separate**: Pre-trained tokenizer, not part of model training
4. **100K vocab problem**: Embedding matrix explodes in size
5. **Multilingual compression**: Chinese uses 2-3x more tokens than English (in English-centric tokenizers)

### Production Insights:
1. **GPT-4**: ~100K vocab, optimized for multilingual + code
2. **Llama 3**: 128K vocab (4x larger than Llama 2) for better multilingual support
3. **Codex**: Code-aware tokenizer (40% fewer tokens for code)
4. **Domain-specific**: BioBERT, LegalBERT (20-40% efficiency gains)

---

## 9. Advanced Topics and Edge Cases

### 9.1 Tokenizer Parallelization and Batching

**Problem**: Tokenizing millions of documents is slow

**Solution**: Parallel tokenization
```python
# HuggingFace Fast Tokenizers (Rust-based)
from tokenizers import Tokenizer

# Single-threaded (slow)
tokens = [tokenizer.encode(text) for text in texts]  # 10 sec

# Batch + parallel (fast)
tokens = tokenizer.encode_batch(texts, is_pretokenized=False)  # 1 sec
# 10x speedup!
```

**Key techniques**:
1. **Batching**: Process 1000+ texts at once
2. **Multi-threading**: Rust tokenizers use all CPU cores
3. **Memory mapping**: Load vocabulary once, share across threads

**Production example (Hugging Face)**:
- Processing 100M documents for training
- Single-threaded: 100 hours
- Parallel (128 cores): 2 hours
- **50x speedup**

### 9.2 Tokenization Artifacts and Failures

**Common issues**:

**1. Whitespace handling**:
```python
# Different tokenizations based on whitespace
"Hello world"   → ["Hello", " world"]
"Hello  world"  → ["Hello", "  world"]  # Different token!
"Hello\tworld"  → ["Hello", "\t", "world"]  # Tab character

# Solution: Normalize whitespace before tokenization
text = " ".join(text.split())
```

**2. Unicode normalization**:
```python
# Same visual, different Unicode
"café" (é = single character U+00E9)
"café" (é = e + combining accent U+0065 U+0301)

# Tokenize differently!
tokenizer.encode("café")  # [1234, 5678]
tokenizer.encode("café")  # [1234, 5679, 5680]

# Solution: Unicode normalization (NFC or NFKC)
import unicodedata
text = unicodedata.normalize("NFKC", text)
```

**3. Leading/trailing spaces**:
```python
# GPT-2 byte-level BPE preserves spaces
"Hello" → ["Hello"]
" Hello" → [" Hello"]  # Different token!

# Can cause issues if not consistent
# Training: Always strip spaces
# Inference: Forget to strip → Different tokens
```

**4. Control characters**:
```python
# Invisible characters can break tokenization
text = "Hello\x00world"  # Null byte
tokens = tokenizer.encode(text)  # May fail or produce garbage

# Solution: Strip control characters
import re
text = re.sub(r'[\x00-\x1F\x7F]', '', text)
```

### 9.3 Tokenization for Multilingual Models

**Challenges by language family**:

**1. East Asian (Chinese, Japanese, Korean)**:
- No spaces between words
- Need word segmentation before tokenization
- Or: Use SentencePiece (language-agnostic)

**Example (Chinese)**:
```python
# Without segmentation
text = "我爱北京天安门"  # "I love Tiananmen Square in Beijing"
tokens = tokenizer.encode(text)
# → 21 tokens (each character = 3 bytes in UTF-8)

# With segmentation (jieba)
import jieba
text = " ".join(jieba.cut("我爱北京天安门"))
# → "我 爱 北京 天安门"
tokens = tokenizer.encode(text)
# → 4 tokens (much better!)
```

**2. Arabic script (Arabic, Persian, Urdu)**:
- Right-to-left (RTL) text
- Diacritics (optional accent marks)
- Contextual letter forms (letters change shape based on position)

**Example**:
```python
# With diacritics
"مَرْحَباً"  # "Hello" (fully vocalized)
# → 15 tokens (each diacritic = separate token)

# Without diacritics
"مرحبا"  # "Hello" (no vocalization)
# → 5 tokens

# Solution: Strip diacritics for tokenization
import arabic_reshaper
text = arabic_reshaper.remove_tashkeel(text)
```

**3. Indic scripts (Hindi, Tamil, Telugu)**:
- Complex conjunct characters
- Combining characters (consonant clusters)

**4. Cyrillic (Russian, Ukrainian, Bulgarian)**:
- Different alphabets, but similar structure to Latin
- Usually works well with standard BPE

### 9.4 Tokenization Cost Analysis

**API pricing based on tokens**:

**OpenAI GPT-4**:
- $0.03 per 1K input tokens (8K context)
- $0.06 per 1K output tokens

**Cost comparison by language**:
```python
# English (4 chars/token)
Document: 10,000 characters (2 pages)
Tokens: 10,000 / 4 = 2,500 tokens
Cost: $0.03 × 2.5 = $0.075

# Chinese (1.5 chars/token with GPT-2 tokenizer)
Document: 10,000 characters (2 pages)
Tokens: 10,000 / 1.5 = 6,667 tokens
Cost: $0.03 × 6.67 = $0.20 (2.7x more expensive!)

# Chinese (2.5 chars/token with GPT-4 improved tokenizer)
Document: 10,000 characters
Tokens: 10,000 / 2.5 = 4,000 tokens
Cost: $0.03 × 4 = $0.12 (1.6x more expensive)
```

**Optimization strategies**:
1. **Use models with better multilingual support** (GPT-4, Llama 3)
2. **Compress text before sending** (remove redundancy)
3. **Cache common prompts** (avoid re-tokenizing)
4. **Batch requests** (amortize fixed costs)

### 9.5 Tokenization in Fine-Tuning

**Critical consideration**: Use the SAME tokenizer as pre-training

**Mistake (common in beginners)**:
```python
# Pre-training: GPT-2 tokenizer (50K vocab)
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Fine-tuning: Train new tokenizer on domain corpus
tokenizer_new = train_tokenizer(medical_corpus)  # 60K vocab

# Fine-tune model with new tokenizer
model.resize_token_embeddings(len(tokenizer_new))
# ❌ BAD: Embeddings for new tokens are random!
```

**Correct approach**:
```python
# Option 1: Use original tokenizer (may be inefficient)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# Fine-tune model as-is

# Option 2: Extend vocabulary (rare)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
new_tokens = ["<MEDICAL>", "hydroxychloroquine", ...]
tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))
# Initialize new embeddings from mean of existing
# Fine-tune model (new embeddings will learn)

# Option 3: Train from scratch (expensive)
# Train new tokenizer + model from scratch on domain corpus
```

**Best practice**: Keep original tokenizer unless you have strong reason to change

### 9.6 Subword Regularization

**Concept**: Randomly sample tokenization during training (improves robustness)

**Standard tokenization** (deterministic):
```python
"unbelievable" → ["un", "believ", "able"]  (always same)
```

**Subword regularization** (stochastic):
```python
"unbelievable" → ["un", "believ", "able"]     (prob 0.5)
"unbelievable" → ["un", "believe", "able"]    (prob 0.3)
"unbelievable" → ["unbe", "liev", "able"]     (prob 0.2)

# Sample randomly during training
# Use most likely segmentation during inference
```

**Benefits**:
1. **Robustness**: Model sees multiple segmentations, generalizes better
2. **Handles typos**: "unbelivable" (typo) similar to training variations
3. **Better rare words**: Multiple segmentations provide more training signal

**Implemented in**: Unigram LM (SentencePiece), not in standard BPE

**Used by**: T5, mT5, XLM-RoBERTa (during pre-training)

### 9.7 Morphological Tokenization

**Problem**: BPE ignores morphology (word structure)

**Example**:
```python
"playing" → ["play", "ing"]  (Good! Captures morpheme)
"player"  → ["play", "er"]   (Good!)
"replay"  → ["rep", "lay"]   (Bad! Should be ["re", "play"])
```

**Morphology-aware tokenization**:
- Use linguistic knowledge (prefixes, suffixes, roots)
- Example: "replay" = prefix "re" + root "play"

**Implementations**:
1. **Rule-based**: Define prefix/suffix rules (labor-intensive)
2. **Unsupervised**: Morfessor (learns morphology from data)
3. **Hybrid**: BPE with morphology constraints

**Trade-off**:
- Better linguistic correctness
- More complex, slower training
- Not widely used in production (BPE is good enough)

### 9.8 Tokenization for Retrieval (RAG)

**Challenge**: Semantic search requires consistent tokenization

**Setup**:
```python
# Indexing: Encode documents
docs = ["Machine learning is...", "Deep learning uses..."]
doc_embeddings = embed_model.encode(docs)
# Store in vector DB

# Query: Encode user question
query = "What is ML?"
query_embedding = embed_model.encode(query)
# Find similar docs (cosine similarity)
```

**Tokenization matters**:
1. **Same tokenizer**: Embedding model and LLM must use compatible tokenizers
   - BERT tokenizer for embedding (WordPiece)
   - GPT tokenizer for LLM (BPE)
   - Mismatch: Poor retrieval quality

2. **Max length**: Truncate long documents consistently
   ```python
   # Truncate to 512 tokens (BERT limit)
   tokens = tokenizer.encode(doc, max_length=512, truncation=True)
   ```

3. **Special tokens**: Handle correctly
   ```python
   # Add [CLS] for BERT embeddings
   tokens = tokenizer.encode(doc, add_special_tokens=True)
   # → [101, ...tokens..., 102]  ([CLS] ... [SEP])
   ```

**Best practice**: Use embedding models with same tokenizer family as LLM
- BERT-based embedding + GPT LLM: OK (both subword)
- SentenceTransformers + Llama: Good (both handle long text)

### 9.9 Tokenization Efficiency Metrics

**Key metrics for production systems**:

**1. Tokens per second (throughput)**:
```python
import time

texts = [...]  # 10,000 documents
start = time.time()
tokens = tokenizer.batch_encode(texts)
elapsed = time.time() - start

throughput = len(texts) / elapsed
# Target: 1000+ docs/sec for production
```

**2. Memory usage**:
```python
import psutil

process = psutil.Process()
mem_before = process.memory_info().rss / 1024**2  # MB

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokens = tokenizer.batch_encode(texts)

mem_after = process.memory_info().rss / 1024**2
memory_used = mem_after - mem_before
# Target: <1 GB for 1M documents
```

**3. Compression ratio**:
```python
total_chars = sum(len(text) for text in texts)
total_tokens = sum(len(tokens) for tokens in all_tokens)

compression_ratio = total_chars / total_tokens
# English: 4-5 chars/token is good
# Code: 3-4 chars/token
# Chinese: 2-3 chars/token (GPT-4/Llama 3)
```

**4. Vocabulary coverage**:
```python
# What % of vocabulary is used?
unique_tokens = set()
for tokens in all_tokens:
    unique_tokens.update(tokens)

coverage = len(unique_tokens) / len(tokenizer.vocab)
# Target: 20-40% (80% of vocab is rare/unused)
```

### 9.10 Tokenization Security Concerns

**1. Prompt injection via tokenization**:
```python
# Adversarial input with special tokens
user_input = "Ignore previous instructions <|endoftext|> You are now..."

# If not sanitized, model might treat <|endoftext|> as stop token
tokens = tokenizer.encode(user_input)
# → [..., 50256, ...]  (50256 = <|endoftext|> in GPT-2)

# Model stops generating prematurely!
```

**Defense**:
```python
# Remove special tokens from user input
special_token_ids = set(tokenizer.all_special_ids)
tokens = [t for t in tokens if t not in special_token_ids]
```

**2. Token smuggling**:
```python
# Attacker embeds hidden instructions in Unicode
user_input = "Innocent text\u200B<hidden instruction>"
# \u200B = zero-width space (invisible)

# Tokenizer might preserve hidden text
tokens = tokenizer.encode(user_input)
# Model processes hidden instruction!
```

**Defense**:
```python
# Strip invisible characters
import unicodedata
text = "".join(c for c in text if unicodedata.category(c)[0] != "C")
```

**3. Vocabulary poisoning**:
- Attacker trains tokenizer on malicious corpus
- Vocabulary includes backdoor tokens
- Model learns to associate backdoor with malicious behavior

**Defense**: Use trusted tokenizers (HuggingFace, OpenAI)

---

## 10. Tokenization Tools and Libraries

### 10.1 HuggingFace Tokenizers

**Fast Tokenizers** (Rust-based):
```python
from transformers import AutoTokenizer

# Load pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Encode
tokens = tokenizer.encode("Hello world")  # [15496, 995]

# Decode
text = tokenizer.decode(tokens)  # "Hello world"

# Batch encode (fast)
tokens = tokenizer.batch_encode_plus(
    ["Hello", "Hi there"],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)
```

**Features**:
- 10x faster than Python tokenizers
- Parallel processing (multi-core)
- Alignment tracking (token → char positions)

### 10.2 SentencePiece

**Google's language-agnostic tokenizer**:
```python
import sentencepiece as spm

# Train tokenizer
spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=32000,
    model_type="bpe",  # or "unigram"
    character_coverage=0.9995
)

# Load and use
sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
tokens = sp.encode("Hello world", out_type=str)  # ["▁Hello", "▁world"]
ids = sp.encode("Hello world", out_type=int)     # [123, 456]
text = sp.decode(ids)                             # "Hello world"
```

**Used by**: T5, Llama, Mistral, Gemma, most multilingual models

### 10.3 tiktoken (OpenAI)

**Fast BPE tokenizer for GPT models**:
```python
import tiktoken

# Load GPT-4 tokenizer
enc = tiktoken.get_encoding("cl100k_base")

# Encode
tokens = enc.encode("Hello world")  # [9906, 1917]

# Decode
text = enc.decode(tokens)  # "Hello world"

# Count tokens (useful for API pricing)
num_tokens = len(enc.encode(text))
```

**Features**:
- Written in Rust (fast)
- Used by GPT-3.5, GPT-4
- No special tokens in vocabulary (handled separately)

### 10.4 minbpe (Karpathy)

**Educational BPE implementation** (for learning):
```python
from minbpe import BasicTokenizer

# Train tokenizer
tokenizer = BasicTokenizer()
tokenizer.train(text, vocab_size=512)

# Encode/decode
tokens = tokenizer.encode("Hello")
text = tokenizer.decode(tokens)
```

**Purpose**: Understanding BPE internals (not for production)

**Source**: github.com/karpathy/minbpe

### 10.5 Comparison Table

| Library | Speed | Language | Use Case |
|---------|-------|----------|----------|
| HF Tokenizers | ⚡⚡⚡ Fast | Rust + Python | Production (general) |
| SentencePiece | ⚡⚡ Medium | C++ + Python | Multilingual, training |
| tiktoken | ⚡⚡⚡ Fast | Rust + Python | OpenAI models |
| minbpe | ⚡ Slow | Python | Learning/education |
| NLTK | ⚡ Slow | Python | Research (old) |

---

## 11. Real-World Debugging Scenarios

### Scenario 1: Model Outputs Gibberish After Fine-Tuning

**Symptoms**:
```python
# Fine-tuned model generates
"Hello ����� world ��"  # Gibberish characters
```

**Root cause**: Tokenizer mismatch

**Debugging steps**:
```python
# 1. Check if tokenizer matches model
print(tokenizer.vocab_size)  # 50257 (GPT-2)
print(model.config.vocab_size)  # 32000 (Llama!)
# Mismatch! ❌

# 2. Verify tokenization roundtrip
text = "Hello world"
tokens = tokenizer.encode(text)
decoded = tokenizer.decode(tokens)
print(decoded)  # Should equal original text
# If not, tokenizer is broken

# 3. Check token IDs in range
print(max(tokens))  # 15496
print(tokenizer.vocab_size)  # 50257
# If max(tokens) >= vocab_size, out of range!
```

**Solution**: Load correct tokenizer
```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModel.from_pretrained("gpt2")
# Both from same source ✅
```

### Scenario 2: Non-English Text Uses 10x More Tokens

**Symptoms**:
```python
english_text = "Hello world" (11 chars)
english_tokens = tokenizer.encode(english_text)  # 2 tokens

chinese_text = "你好世界" (4 chars)
chinese_tokens = tokenizer.encode(chinese_text)  # 12 tokens (!!)
# 4 chars → 12 tokens = 3 tokens per character
```

**Root cause**: English-centric tokenizer (GPT-2, GPT-3)

**Why it happens**:
- Byte-level BPE trained on English corpus
- Chinese characters rare → decomposed to UTF-8 bytes
- Each Chinese character = 3 bytes in UTF-8
- Result: 3 tokens per character

**Solution 1**: Use multilingual tokenizer
```python
# Llama 3 (128K vocab, trained on 100+ languages)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B")
tokens = tokenizer.encode("你好世界")  # 2-4 tokens ✅
```

**Solution 2**: Pre-tokenize with segmenter
```python
import jieba
chinese_text = "你好世界"
segmented = " ".join(jieba.cut(chinese_text))  # "你好 世界"
tokens = tokenizer.encode(segmented)  # Better efficiency
```

### Scenario 3: Padding Breaks Attention

**Symptoms**:
```python
# Batch with padding
inputs = tokenizer(
    ["Hello", "Hello world from the universe"],
    padding=True,
    return_tensors="pt"
)
# Batch shape: [2, 6] (padded to longest)
# → [[15496, 0, 0, 0, 0, 0],
#    [15496, 995, 422, 262, 6881, 0]]

outputs = model(**inputs)
# Wrong! Model attends to padding tokens ❌
```

**Root cause**: Missing attention mask

**Solution**:
```python
inputs = tokenizer(
    ["Hello", "Hello world from the universe"],
    padding=True,
    return_tensors="pt"
)
# inputs now contains "attention_mask"
# → [[1, 0, 0, 0, 0, 0],
#    [1, 1, 1, 1, 1, 0]]

outputs = model(**inputs)  # ✅ Padding ignored
```

### Scenario 4: Context Overflow

**Symptoms**:
```python
# User provides long document
document = "..." (50,000 characters)
tokens = tokenizer.encode(document)
print(len(tokens))  # 12,500 tokens

# Model context limit: 4,096 tokens
outputs = model(input_ids=tokens)
# Error: "Token indices sequence length is longer than maximum (12500 > 4096)"
```

**Solution 1**: Truncate
```python
tokens = tokenizer.encode(
    document,
    max_length=4096,
    truncation=True
)
# First 4,096 tokens only (loses information!)
```

**Solution 2**: Sliding window
```python
max_length = 4096
stride = 2048  # 50% overlap

chunks = []
for i in range(0, len(tokens), stride):
    chunk = tokens[i:i + max_length]
    if len(chunk) < max_length:
        break
    chunks.append(chunk)

# Process each chunk separately
outputs = [model(input_ids=chunk) for chunk in chunks]
# Aggregate results
```

**Solution 3**: Summarization
```python
# Summarize document first (reduces length)
summary = summarization_model(document)
tokens = tokenizer.encode(summary)  # Much shorter
outputs = model(input_ids=tokens)
```

### Scenario 5: Inconsistent Results Across Runs

**Symptoms**:
```python
# Run 1
text = "Hello world"
tokens1 = tokenizer.encode(text)  # [15496, 995]

# Run 2 (different machine, same code)
tokens2 = tokenizer.encode(text)  # [15496, 995, 0]
# Extra token! ❌
```

**Root cause**: Tokenizer configuration mismatch

**Debugging**:
```python
# Check tokenizer settings
print(tokenizer.padding_side)    # "right" vs "left"
print(tokenizer.add_special_tokens)  # True vs False
print(tokenizer.truncation_side)  # "right" vs "left"

# Different settings → different tokenization
```

**Solution**: Save/load tokenizer configuration
```python
# Save (training)
tokenizer.save_pretrained("my_model/")

# Load (inference)
tokenizer = AutoTokenizer.from_pretrained("my_model/")
# Guarantees same settings ✅
```

---

## 12. Interview Questions - Advanced

### Q26: How would you design a custom tokenizer for a new language?

**Expected answer**:

**Step 1: Corpus collection**
- Gather 10GB+ representative text
- Diverse sources: news, books, web, social media
- Quality filter: Remove spam, duplicates

**Step 2: Character analysis**
- Analyze character frequency distribution
- Identify common characters (should be in base vocab)
- Handle special cases (emojis, punctuation)

**Step 3: Pre-tokenization rules**
- Define word boundaries (spaces, punctuation)
- Handle language-specific rules:
  - Chinese: Use segmenter (jieba, pkuseg)
  - Arabic: Strip diacritics or preserve?
  - German: Split compounds or not?

**Step 4: Algorithm selection**
- **BPE**: Simple, effective for most languages
- **Unigram LM**: Better for morphologically rich languages (Turkish, Finnish)
- **SentencePiece**: Language-agnostic, no pre-tokenization

**Step 5: Training**
```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_lang",
    vocab_size=32000,
    character_coverage=0.9995,  # Cover 99.95% of characters
    model_type="bpe"
)
```

**Step 6: Evaluation**
- Compression ratio: Chars per token
- Coverage: % of test set tokenized without UNK
- Linguistic validity: Does tokenization respect morphemes?

**Step 7: Iteration**
- Test on downstream tasks (classification, generation)
- Adjust vocab size, pre-tokenization rules
- Re-train if necessary

### Q27: Explain how you would migrate from one tokenizer to another in production.

**Expected answer**:

**Challenge**: Can't change tokenizer without retraining model (embeddings are tied to vocab)

**Options**:

**Option 1: Full retrain** (ideal but expensive)
```
1. Train new tokenizer on full corpus
2. Re-tokenize all training data
3. Train model from scratch with new tokenizer
4. Deploy new model

Cost: Weeks-months, $100K+ in compute
Benefit: Clean migration, optimal tokenizer
```

**Option 2: Continue pre-training** (faster)
```
1. Train new tokenizer
2. Initialize new embeddings:
   - Map old tokens to new tokens (if overlap)
   - Initialize unmapped tokens randomly or from mean
3. Continue pre-training on subset of data (10%)
4. Fine-tune on downstream tasks
5. Deploy

Cost: Days-weeks, $10K in compute
Benefit: Preserves some learned knowledge
```

**Option 3: Vocabulary extension** (least disruptive)
```
1. Keep old tokenizer as base
2. Add new tokens to vocabulary (domain-specific)
3. Extend embedding matrix
4. Fine-tune on domain data
5. Deploy

Cost: Hours-days, minimal compute
Benefit: Minimal disruption
Limitation: Only adds tokens, doesn't remove inefficiencies
```

**Production strategy** (blue-green deployment):
1. Deploy new model (Model B) alongside old (Model A)
2. Route 10% traffic to Model B (shadow mode)
3. Compare metrics: Quality, latency, cost
4. Gradually increase traffic: 10% → 50% → 100%
5. Deprecate Model A after validation

### Q28: How does tokenization affect model interpretability?

**Expected answer**:

**Challenge**: Subword tokenization breaks words, making interpretation harder

**Example**:
```python
# Input sentence
"The model is unbelievable"

# Tokenization
["The", " model", " is", " un", "believ", "able"]

# Attention weights
# How do we interpret attention to "un" vs "believable"?
# They're part of the same word but separate tokens!
```

**Impact on interpretability methods**:

**1. Attention visualization**:
```python
# Token-level attention
[The → un]: 0.1      # What does this mean?
[The → believ]: 0.2  # Hard to interpret subwords
[The → able]: 0.05

# Better: Aggregate subword attentions
[The → unbelievable]: 0.1 + 0.2 + 0.05 = 0.35
```

**2. Feature attribution (SHAP, LIME)**:
```python
# SHAP value per token
"un": +0.3      # Contributes positively
"believ": -0.1  # Contributes negatively
"able": +0.2    # Contributes positively

# Interpretation: "unbelievable" overall contribution = 0.3 - 0.1 + 0.2 = 0.4
# But individual subwords are hard to interpret
```

**Solutions**:

**1. Detokenize before interpretation**:
```python
# Map token attributions back to words
tokens = ["un", "believ", "able"]
word = "unbelievable"
attribution = sum(attr["un"], attr["believ"], attr["able"])
# Show word-level attribution
```

**2. Use character-level models** (for interpretability-critical apps):
- Each character is a token
- Perfect word alignment
- But: Much slower, longer sequences

**3. Post-hoc word alignment**:
```python
# Track token → character mapping during tokenization
tokenizer.encode("unbelievable", return_offsets_mapping=True)
# → [("un", (0, 2)), ("believ", (2, 8)), ("able", (8, 12))]
# Use offsets to map tokens back to words
```

### Q29: Design a tokenization strategy for a code generation model.

**Expected answer**:

**Requirements**:
- Handle multiple programming languages
- Preserve syntax structure
- Efficient compression
- Support code-specific patterns (variable names, operators)

**Design decisions**:

**1. Base tokenizer**: Byte-level BPE
- Handles any language (Python, JavaScript, C++, etc.)
- No UNK tokens (fallback to bytes)

**2. Pre-tokenization** (critical for code):
```python
# Language-agnostic rules
1. Split on syntax: (, ), {, }, [, ], ;, :
2. Keep operators as single tokens: ==, !=, <=, >=, ++, --
3. Split strings: "hello world" → [", hello world, "]
4. Preserve indentation: 4 spaces → <INDENT> token
5. Camel case: getUserName → [get, User, Name]
6. Snake case: get_user_name → [get_, user_, name]
```

**3. Vocabulary construction**:
```python
# Code-specific tokens to include
- Keywords: def, class, if, else, for, while, return
- Operators: ==, !=, ++, --, <=, >=, &&, ||, ::
- Common patterns: (), [], {}, <>, "", ''
- Indentation: <INDENT>, <DEDENT> (Python)
- Comments: //, /*, */, #

# Train on code corpus
- GitHub: 100GB of code (Python, JavaScript, Java, C++, Go)
- StackOverflow: Code snippets
- Documentation: Code examples
```

**4. Multi-language handling**:
```python
# Option 1: Single vocabulary (all languages)
vocab_size = 50000
# Pros: Shared patterns (if, for, etc.)
# Cons: May not optimize for any specific language

# Option 2: Language-specific vocabularies
vocab_python = 32000
vocab_javascript = 32000
# Pros: Optimized per language
# Cons: Need to switch tokenizer based on language

# Recommendation: Option 1 (Codex approach)
# Most code patterns are language-agnostic
```

**5. Special handling**:
```python
# Variable names: Preserve as single token if short
getUserName → [get, User, Name]  # Split if long
x → [x]  # Keep if short (1-3 chars)

# Function signatures
def get_user(id: int) -> User:
→ [def, get_user, (, id, :, int, ), ->, User, :]
# Each syntax element is a token

# Nested structures: Brackets balanced
{{{ code }}}
→ [{, {, {, code, }, }, }]
# Tokenizer preserves structure
```

**6. Evaluation metrics**:
```python
# Compression ratio
Code: 10,000 characters
Tokens: 2,500 tokens
Ratio: 4 chars/token (target: 3-5)

# Syntax preservation
Parse tokenized code → Should parse correctly
If syntax errors after detokenization → Bad tokenizer

# Downstream performance
Code completion: Accuracy, speed
Code generation: Correctness (pass@k on HumanEval)
```

**Real-world example** (Codex):
- Vocabulary: 50K tokens
- 40% are code-specific (operators, keywords, patterns)
- Compression: 3.5 chars/token (vs 4-5 for GPT-3 on text)
- Result: 30% fewer tokens for code

### Q30: How would you debug a tokenization-related memory leak in production?

**Expected answer**:

**Symptoms**:
```python
# Memory usage grows over time
Hour 1: 4 GB
Hour 2: 6 GB
Hour 3: 8 GB
Hour 4: OOM crash (Out of Memory)
```

**Debugging process**:

**Step 1: Profile memory**
```python
import tracemalloc
import gc

tracemalloc.start()

# Tokenize 10,000 documents
for doc in documents:
    tokens = tokenizer.encode(doc)
    # Process tokens
    
current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024**2:.2f} MB")
print(f"Peak: {peak / 1024**2:.2f} MB")

# Check if memory is released
gc.collect()
current_after = tracemalloc.get_traced_memory()[0]
print(f"After GC: {current_after / 1024**2:.2f} MB")
```

**Step 2: Common causes**

**Cause 1: Tokenizer cache growing unbounded**
```python
# Many tokenizers cache tokenization results
tokenizer._tokenize_cache  # Dictionary grows infinitely

# Fix: Disable cache or limit size
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer._tokenize_cache = {}  # Clear cache
# Or set max cache size
```

**Cause 2: Not releasing token tensors**
```python
# Accumulating tensors in memory
all_tokens = []
for doc in documents:
    tokens = tokenizer.encode(doc, return_tensors="pt")
    all_tokens.append(tokens)  # ❌ Keeps tensors in memory

# Fix: Process and release
for doc in documents:
    tokens = tokenizer.encode(doc, return_tensors="pt")
    output = model(tokens)
    # Process output
    del tokens, output  # Explicitly delete
    torch.cuda.empty_cache()  # If using GPU
```

**Cause 3: Circular references**
```python
# Tokenizer holds references to input strings
class CustomTokenizer:
    def __init__(self):
        self.cache = {}  # Stores original strings
    
    def encode(self, text):
        if text in self.cache:
            return self.cache[text]
        tokens = self._tokenize(text)
        self.cache[text] = tokens  # ❌ Holds string forever
        return tokens

# Fix: Use weak references or limit cache size
from collections import OrderedDict

class CustomTokenizer:
    def __init__(self, max_cache=10000):
        self.cache = OrderedDict()
        self.max_cache = max_cache
    
    def encode(self, text):
        if text in self.cache:
            return self.cache[text]
        tokens = self._tokenize(text)
        self.cache[text] = tokens
        if len(self.cache) > self.max_cache:
            self.cache.popitem(last=False)  # Remove oldest
        return tokens
```

**Step 3: Monitor in production**
```python
import psutil

def monitor_memory():
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024**2
    
    if mem_mb > 8000:  # Alert if > 8 GB
        logger.warning(f"High memory: {mem_mb:.2f} MB")
        # Trigger garbage collection
        gc.collect()
        torch.cuda.empty_cache()
    
    return mem_mb

# Call periodically
mem_usage = monitor_memory()
```

**Step 4: Fix and validate**
```python
# Before fix
memory_usage = []
for i in range(100):
    tokens = tokenizer.encode(documents[i])
    memory_usage.append(monitor_memory())

# memory_usage should be flat, not increasing
# If increasing → leak exists

# After fix
# Re-run test, verify memory is stable
```

---

## 13. Key Resources and References

### Essential Papers
1. **"Neural Machine Translation of Rare Words with Subword Units"** (Sennrich et al., 2016)
   - Original BPE for NLP paper
   - Link: arxiv.org/abs/1508.07909

2. **"SentencePiece: A simple and language independent approach to subword tokenization"** (Kudo & Richardson, 2018)
   - Google's language-agnostic tokenizer
   - Link: arxiv.org/abs/1808.06226

3. **"Language Models are Unsupervised Multitask Learners"** (GPT-2 paper, 2019)
   - Byte-level BPE introduction
   - Link: openai.com/research/gpt-2

4. **"Language Models are Few-Shot Learners"** (GPT-3 paper, 2020)
   - Scaling analysis, vocabulary decisions
   - Link: arxiv.org/abs/2005.14165

### Practical Guides
1. **HuggingFace Tokenizers documentation**
   - huggingface.co/docs/tokenizers
   - Comprehensive guide to training and using tokenizers

2. **Andrej Karpathy's minbpe**
   - github.com/karpathy/minbpe
   - Educational BPE implementation with explanations

3. **SentencePiece GitHub**
   - github.com/google/sentencepiece
   - Official implementation and tutorials

4. **OpenAI tiktoken**
   - github.com/openai/tiktoken
   - Fast BPE tokenizer for GPT models

### Blog Posts
1. **"Let's build the GPT Tokenizer"** (Andrej Karpathy)
   - youtube.com/watch?v=zduSFxRajkE
   - Video walkthrough of BPE implementation

2. **"How to train a new language model from scratch using Transformers and Tokenizers"** (HuggingFace)
   - huggingface.co/blog/how-to-train

3. **"The Tokenization Problem"** (Chip Huyen)
   - huyenchip.com/2023/04/11/llm-engineering.html
   - Discusses tokenization challenges in production

### Tools
1. **HuggingFace Tokenizers** (Rust-based, fast)
2. **SentencePiece** (Google, language-agnostic)
3. **tiktoken** (OpenAI, for GPT models)
4. **minbpe** (Karpathy, educational)

---

## 14. Summary: Complete Interview Checklist

### Core Concepts ✅
- [ ] BPE algorithm (frequency-based merging)
- [ ] Byte-level BPE (UTF-8 bytes, no UNK)
- [ ] WordPiece (likelihood-based)
- [ ] SentencePiece (language-agnostic)
- [ ] Special tokens (PAD, BOS, EOS, UNK, MASK)
- [ ] Vocabulary construction process
- [ ] Training corpus selection

### Trade-offs ✅
- [ ] Vocabulary size vs memory (embedding matrix)
- [ ] Vocabulary size vs speed (softmax complexity)
- [ ] Sequence length vs attention (O(n²))
- [ ] Subword vs word vs character level
- [ ] Compression ratio (chars/token) by language

### Production Skills ✅
- [ ] Training tokenizer from scratch
- [ ] Debugging tokenization issues
- [ ] Handling multilingual text
- [ ] Domain adaptation (BioBERT, CodeBERT)
- [ ] Memory profiling and optimization
- [ ] Tokenization security (prompt injection)

### Calculations ✅
- [ ] Embedding matrix memory: `vocab × embed_dim × bytes`
- [ ] Token count estimation: `chars / compression_ratio`
- [ ] API cost: `tokens × price_per_1k`
- [ ] Attention complexity: `O(seq_len²)`
- [ ] KV cache: `2 × layers × seq_len × ...`

### Real-World Examples ✅
- [ ] GPT-2/3/4 tokenization evolution
- [ ] Llama 2 vs Llama 3 (32K → 128K vocab)
- [ ] Codex code-aware tokenization
- [ ] Multilingual models (mT5, XLM-R)
- [ ] Domain-specific (BioBERT, LegalBERT)

### Common Traps ✅
- [ ] Training-serving skew (different tokenizers)
- [ ] Vocabulary size 100K problem (memory explosion)
- [ ] Multilingual inefficiency (Chinese 3x tokens)
- [ ] Padding without attention masks
- [ ] Context overflow (document too long)
- [ ] Tokenizer cache memory leaks

**You're now ready for any tokenization interview question!** 🎯

---

**Document Status**: ✅ **COMPLETE**  
**Coverage**: All fundamental topics + advanced edge cases + production scenarios  
**Interview Readiness**: 100% - Covers theory, practice, debugging, and real-world
