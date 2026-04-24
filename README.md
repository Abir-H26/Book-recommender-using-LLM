# Semantic Book Recommender

I built this project after following the [freeCodeCamp LLM course](https://www.youtube.com/watch?v=Q7mS1VHm3Yw) by Dr. Jodie Burchell from JetBrains. The idea is simple — instead of browsing through endless lists or relying on star ratings, you just describe the kind of book you're in the mood for and the app finds something that actually matches what you mean.

---

## Why LLMs?

This is honestly the most interesting part of the project for me, so I want to explain it properly.

Most recommendation systems are built around ratings, purchase history, or rigid genre tags. They work fine if you want "more books like this one" but fall apart the moment you try something like *"a quiet story about grief told with dark humour"* — because there's no field in a database for that.

LLMs change this because they don't just match words, they understand meaning.

**Keyword search vs semantic search**
If you search for "family drama", a traditional system returns books where those words appear. A semantic search returns books that *feel* like a family drama, even if the description never uses those exact words. The LLM has read enough text to know what a family drama feels like — tension, generational conflict, complicated relationships — and it finds books that carry those qualities.

**Context matters**
Older models like Word2Vec give every word a single fixed meaning. The word "bank" gets one vector, whether you're talking about a riverbank or a financial institution. Transformer-based LLMs understand the word in context, which makes the representations far richer and much more useful for matching nuanced queries to book descriptions.

**No labelled data needed for classification**
Sorting 5,000+ books into genres by hand would be a nightmare. With a model like `facebook/bart-large-mnli`, I can classify every book as Fiction, Non-Fiction, Children's Fiction, or Children's Non-Fiction without writing a single labelled training example. The model already knows enough about the world to figure it out — it just needs to be asked.

**Emotion as a feature**
This is my favourite part. A normal dataset might tell you a book is "literary fiction". It won't tell you whether it's devastating or uplifting. Because we're working with raw text descriptions, we can run them through a fine-tuned emotion classifier and actually extract that information — joy, fear, sadness, surprise — and use it as a filter. That's something you simply can't do with a spreadsheet of ratings.

The short version: LLMs let the recommender understand a book the way a reader would, rather than the way a database would.

---

## What it does

- Takes a plain English description of the book you're looking for
- Finds the most semantically similar books from a dataset of ~5,200 titles
- Lets you filter by genre: Fiction, Non-Fiction, Children's Fiction, Children's Non-Fiction
- Lets you sort results by emotional tone: Happy, Surprising, Angry, Suspenseful, Sad
- Shows results in a Gradio dashboard with book covers and truncated descriptions

---

## How it works

There are four main steps, each covered in its own notebook.

**1. Data cleaning (`data-exploration.ipynb`)**
The raw data comes from the [7K Books dataset on Kaggle](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata). A lot of the descriptions are too short to be useful — anything under 25 words gets dropped. Books with missing ratings or publication years go too. After cleaning you're left with around 5,200 books, each with a description long enough to actually say something about the book.

Each description also gets its ISBN13 prepended to the front. This sounds odd but it's the cleanest way to link recommendations back to the full book record later — splitting the ISBN off the returned description is much faster and more reliable than trying to do a full-text match.

**2. Vector search (`vector-search.ipynb`)**
Every description gets converted into a vector embedding using OpenAI's Ada model, then stored in a Chroma vector database via LangChain. When someone types a query, that query gets embedded the same way and the database returns whichever book descriptions are closest in vector space. The whole setup is only a handful of lines of LangChain code — it's surprisingly straightforward once you see it.

**3. Genre classification (`text-classification.ipynb`)**
`facebook/bart-large-mnli` handles this with zero-shot classification — no training, no labelled examples, just the model using what it already knows. Accuracy came out around 78% on the labelled examples I had to test against, which is pretty solid for something that required no training data at all.

**4. Emotion extraction (`sentiment-analysis.ipynb`)**
Rather than classifying the whole description as a single emotion, each description gets split into individual sentences and the emotion classifier runs over each one separately. The maximum score per emotion across all sentences is kept for each book. This gives a much more nuanced picture — a book can register high on both joy and sadness, which is often accurate. The model used here is `j-hartmann/emotion-english-distilroberta-base`, a fine-tuned RoBERTa model that covers anger, disgust, fear, joy, sadness, surprise, and neutral.

**5. The dashboard (`gradio-dashboard.py`)**
Everything comes together in a Gradio app. Type a query, pick a category filter if you want one, pick a tone, hit the button, and you get 16 book recommendations displayed as a gallery with cover images.

---

## Getting started

You'll need Python 3.11+ and an OpenAI API key with some credit loaded. The Ada embedding model is cheap — $10 will last a long time.

```bash
git clone https://github.com/t-redactyl/llm-semantic-book-recommender.git
cd llm-semantic-book-recommender
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Download the dataset:

```python
import kagglehub
path = kagglehub.dataset_download("dylanjcastillo/7k-books-with-metadata")
```

Run the notebooks in order:

data-exploration.ipynb
vector-search.ipynb
text-classification.ipynb
sentiment-analysis.ipynb

Then launch the dashboard:

```bash
python gradio-dashboard.py
```

It'll open at `http://127.0.0.1:7860` by default.

---

## Stack

| Package | What it's for |
|---|---|
| `langchain` + `langchain-openai` + `langchain-chroma` | Vector database and embeddings pipeline |
| `transformers` | Running the Hugging Face classification and emotion models |
| `gradio` | The dashboard UI |
| `pandas` / `numpy` | Data wrangling |
| `python-dotenv` | Keeping the API key out of the code |
| `kagglehub` | Downloading the dataset |
| `seaborn` / `matplotlib` | Visualisations during exploration |

---

## References

- Original course and code: [Dr. Jodie Burchell](https://github.com/t-redactyl/llm-semantic-book-recommender)
- [freeCodeCamp course video](https://www.youtube.com/watch?v=Q7mS1VHm3Yw)
- [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course) — genuinely the best free resource for understanding how these models work
- [LangChain docs](https://python.langchain.com/docs/introduction/)
- [Gradio docs](https://gradio.app/guides/quickstart)
