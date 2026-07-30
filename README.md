# Knowledge Base Entity Linking (KBEL)

KBEL is a Python framework for **Knowledge Base Entity Linking**. It identifies candidate entities from one or more knowledge bases and links textual mentions to the most appropriate entity using configurable disambiguation strategies.

The framework is built around three core concepts:

- **KnowledgeSource**: retrieves candidate entities from a knowledge base.
- **Disambiguator**: ranks the retrieved candidates according to a given strategy.
- **KBEL**: orchestrates the entity linking pipeline by combining a knowledge source and an optional disambiguation strategy.

Current knowledge source implementations are built on top of the abstractions provided by [KIF](https://pypi.org/project/kif-lib/), making it easy to integrate multiple knowledge bases while exposing a unified API.

See the complete examples in the [demo notebook](./examples/demo.ipynb).

---

## Features

- Plugin architecture for both knowledge sources and disambiguation strategies.
- Support for multiple knowledge sources.
- Candidate retrieval for both **Items** and **Properties**.
- Embedding-based semantic disambiguation.
- LLM-based disambiguation using OpenAI, LangChain-compatible models, or custom providers.
- Easily extensible through custom plugins.

---

## Available Disambiguation Strategies

KBEL currently provides the following strategies:

- **Naive** (`naive`): Returns the top-ranked candidate retrieved from the knowledge source.
- **Similarity** (`sim`): Ranks candidates according to semantic similarity between the mention context and candidate descriptions.
- **LLM** (`llm`): Uses a Large Language Model to select the most appropriate candidate given the mention and its context.

---

# Installation

```bash
pip install kbel
```

---

# Quick Start

```python
from kbel import KBEL
from kbel.knowledge_sources import KnowledgeSource
from kbel.disambiguators import Disambiguator
from kbel.core import Mention

ks = KnowledgeSource("wikidata")

kbel = KBEL(ks)
kbel.disambiguator = Disambiguator("naive")

mention = Mention(
    label="Python",
    text="Python is used for coding.",
    entity_type=EntityType.ITEM
)
results = kbel.disambiguate(mention)

for label, description, entity in results:
    print(label, description, entity)
```

---

# Similarity Disambiguator

```python
kbel.disambiguator = Disambiguator("sim")

results = kbel.disambiguate(mention)
```

---

# LLM Disambiguator

Using OpenAI models through LangChain:

```python
from langchain_openai import ChatOpenAI
import os

model = ChatOpenAI(
    model="gpt-5.2",
    api_key=os.environ["LLM_API_KEY"]
)

kbel.disambiguator = Disambiguator(
    "llm",
    model=model
)

results = kbel.disambiguate(
    Mention(
        label="Python",
        text="Python is used for coding.",
        entity_type=EntityType.ITEM,
        context="""
        Python is a high-level, general-purpose programming language that emphasizes code readability, simplicity, and ease-of-writing with the use of significant indentation,[38] an extensive ("batteries-included") standard library, and garbage collection. Python supports multiple programming paradigms but with an emphasis on object-oriented programming and dynamic typing.
        """
    )
)

for result in results:
    print(result)
```
---

## Changing the Knowledge Source

The same disambiguation strategy can be used with different knowledge bases.

```python
kbel.ks = KnowledgeSource("dbpedia", limit=10)

results = kbel.disambiguate(mention)

for result in results:
    print(result)
```

---


# Extending KBEL

KBEL is **extensible**: New plugins can be added by subclassing:

- `KnowledgeSource`
- `Disambiguator`

Both components are automatically registered and can be instantiated by name.

---


## License

Released under the [Apache-2.0 license](./LICENSE).