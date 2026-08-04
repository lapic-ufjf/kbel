# Knowledge Base Entity Linking (KBEL)

KBEL is a Python framework for **Knowledge Base Entity Linking**. It identifies candidate entities from one or more knowledge bases and links textual mentions to the most appropriate entity using configurable disambiguation strategies.

The framework is built around three core concepts:

- **Search**: retrieve candidate entities from a knowledge base.
- **Disambiguator**: ranks the retrieved candidates according to a given strategy.
- **KBEL**: orchestrates the entity linking pipeline.

Current `Search` implementations are built on top of the abstractions provided by [KIF](https://pypi.org/project/kif-lib/), making it easy to integrate multiple knowledge bases while exposing a unified API.

See the complete examples in the [demo notebook](./examples/demo.ipynb).

---

## Features

- Plugin architecture for both knowledge bases and disambiguation strategies.
- Support for multiple knowledge bases.
- Candidate retrieval for both **Items** and **Properties**.
- Embedding-based semantic disambiguation.
- LLM-based disambiguation using OpenAI, LangChain-compatible models, or custom providers.
- Easily extensible through custom plugins.

---

## Available Disambiguation Strategies

KBEL currently provides the following strategies:

- **Naive** (`naive`): Returns the top-ranked candidate retrieved from the knowledge base.
- **Similarity** (`sim`): Ranks candidates according to semantic similarity between the mention context and candidate descriptions.
- **LLM** (`llm`): Uses a Large Language Model to select the most appropriate candidate given the mention and its context.

---

# Installation

Install the core package:

```bash
pip install kbel
```

Additional dependencies are required for some disambiguation strategies:

| Strategy | Installation |
|----------|--------------|
| Naive | Included in the base installation |
| Similarity | `pip install "kbel[similarity]"` |
| LLM | `pip install "kbel[llm]"` |

If the required optional dependencies are not installed, an `ImportError` will be raised when attempting to instantiate the corresponding disambiguator.

---

# Quick Start

```python
from kbel import KBEL
from kbel.disambiguators import Disambiguator
from kbel.core import Mention, EntityType

kbel = KBEL("wikidata")

from kbel.disambiguators.naive import NaiveDisambiguator
kbel.disambiguator = Disambiguator("naive")

mention = Mention(
    label="Python",
    text="Python is used for coding.",
    entity_type=EntityType.ITEM,
)

results = kbel.link(mention)

for label, description, entity in results:
    print(label, description, entity)
```

---

# Similarity Disambiguator

> **Note:** Install the optional dependencies first:
>
> ```bash
> pip install "kbel[similarity]"
> ```

```python
from kbel.disambiguators.similarity import SimilarityDisambiguator
kbel.disambiguator = Disambiguator("sim")

results = kbel.link(mention)

for result in results:
    print(result)
```

---

# LLM Disambiguator

> **Note:** Install the optional dependencies first:
>
> ```bash
> pip install "kbel[llm]"
> ```

Using OpenAI models through LangChain:

```python
from langchain_openai import ChatOpenAI
import os

model = ChatOpenAI(
    model="gpt-5.2",
    api_key=os.environ["LLM_API_KEY"]
)

from kbel.disambiguators.llm import LLM_Disambiguator
kbel.disambiguator = Disambiguator(
    "llm",
    model=model
)

results = kbel.link(
    Mention(
        label="Python",
        text="Python is used for coding.",
        entity_type=EntityType.ITEM,
        context="""
        Python is a high-level, general-purpose programming language that emphasizes code readability, simplicity, and ease of writing through significant indentation, an extensive standard library, and automatic memory management. Python supports multiple programming paradigms, with a strong emphasis on object-oriented programming.
        """
    )
)

for result in results:
    print(result)
```

---

## Changing the Knowledge Base

The same disambiguation strategy can be used with different knowledge bases.

```python
kbel.knowledge_base= "dbpedia"

results = kbel.link(mention)

for result in results:
    print(result)
```

---

# Extending KBEL

KBEL is **extensible**. New plugins can be added by subclassing:

- `Disambiguator`
- `Search` from `kif_lib`


Both components are automatically registered and can be instantiated by name.

---

## License

Released under the [Apache-2.0 license](./LICENSE).