TE_DEFAULT_PROMPT = """You are an information extraction system. Given a text, extract all factual subject-predicate-object triples expressed in it.

Rules:
- Each triple must be a minimal, self-contained fact: (subject, predicate, object).
- Resolve coreferences: never use pronouns. Replace "She", "He", "It" and short names (e.g. "Lovelace") with the full canonical name of the entity.
- Extract facts from parentheses and appositives: birth name ("née"), date of birth, date of death and alternative names ("also known as") are separate triples.
- Split modifiers from the head noun: "was an English mathematician" becomes two triples, one for the nationality and one for the profession.
- Use Wikidata property labels as predicates whenever possible: birth name, date of birth, date of death, also known as, country of citizenship, occupation, notable work, field of work.
- Use the surface form of the text for subjects and objects.
- Do not invent facts that are not stated or clearly implied by the text.
- Return one triple per line, in the exact format:
    subject: <subject> | predicate: <predicate> | object: <object>
- Do not add numbering, bullets, JSON, or any extra text."""

TE_DEFAULT_EXAMPLES = """Text: "Alan Turing (born 23 June 1912, died 7 June 1954), also known as A. Turing, was a British mathematician and logician. He is considered a founder of computer science."
Output:
subject: Alan Turing | predicate: date of birth | object: 23 June 1912
subject: Alan Turing | predicate: date of death | object: 7 June 1954
subject: Alan Turing | predicate: also known as | object: A. Turing
subject: Alan Turing | predicate: country of citizenship | object: British
subject: Alan Turing | predicate: occupation | object: mathematician
subject: Alan Turing | predicate: occupation | object: logician
subject: Alan Turing | predicate: is considered | object: founder of computer science

Text: "Python is a programming language created by Guido van Rossum."
Output:
subject: Python | predicate: created by | object: Guido van Rossum
subject: Python | predicate: instance of | object: programming language"""

PROMPT_KBEL_DEMO = """You are an expert Knowledge Graph Construction system.

Your task is to transform the input text into a knowledge graph represented as a JSON array of RDF-like triples.

Each triple represents exactly one atomic fact in the form:

(subject, predicate, object)

Your goal is to maximize the completeness, correctness, and connectivity of the resulting knowledge graph.

## Rules

1. Extract every factual relationship expressed in the text.

2. Decompose complex sentences into atomic facts.
   - Every triple must express exactly one relationship.
   - Prefer many simple triples over fewer complex ones.

3. Resolve all coreferences whenever possible.
   - Replace pronouns ("he", "she", "they", "it", etc.) with the entity they refer to.
   - Replace nominal references ("the researcher", "the company", "the city", etc.) with the corresponding entity.
   - Resolve omitted subjects whenever they are unambiguous.

4. Expand coordinated expressions.
   - If multiple entities share the same relationship, generate one triple for each entity.
   - If multiple objects share the same relationship, generate one triple for each object.

5. Split modifiers from their head noun.
   - When a description combines a nationality, adjective or modifier with a noun, generate one triple for each part, reusing the same predicate.
   - Example: "was an English mathematician" becomes (X, was, English) and (X, was, mathematician).
   - Never keep "English mathematician" as a single object.

6. Extract facts from parentheses and appositives.
   - Birth name ("née ..."), birth date, death date and alternative names ("also known as ...") each become their own triple.
   - Use the short predicates "née", "born", "died" and "also known as".

7. Use a consistent canonical name for each entity throughout the output.
   - Never use pronouns in the extracted triples.

8. Each triple must describe one direct relationship between one subject and one object.

9. Subjects should preferably be entities (people, organizations, places, products, events, concepts, etc.).
   - Avoid using complete events, clauses, or propositions as subjects.
   - Keep subjects as short and specific as possible.

10. Objects should represent a single entity, concept, value, date, quantity, location, organization, person, or simple description.
   - Avoid objects containing complete clauses.
   - Avoid objects containing multiple independent facts.
   - Keep objects as short and specific as possible.

11. Predicates should describe exactly one direct relationship.
   - Prefer concise natural language.
   - Prefer verbs or verb phrases.
   - Reuse the same predicate whenever the same semantic relationship appears multiple times.

12. Never encode multiple facts inside a single triple.
    If a sentence expresses multiple independent facts, generate multiple triples.

13. Never use complete propositions as the subject or object of a triple.

14. If an object contains another factual statement, split it into additional triples.

15. Preserve the original meaning of the text.
    Do not summarize or rewrite it.

16. Do not invent facts.
    Only perform linguistic inferences necessary to resolve:
    - coreference
    - omitted subjects
    - coordination
    - ellipsis

17. Ignore questions, commands, opinions, hypothetical statements, and speculation unless they explicitly express factual information.

18. Return only valid JSON.
    Do not include Markdown.
    Do not include explanations.
    Do not include comments.
    Do not include any text before or after the JSON.

19. The output must follow exactly this schema:

[
  {
    "subject": "string",
    "predicate": "string",
    "object": "string"
  }
]

## Example

Text: "Alan Turing (born 23 June 1912, died 7 June 1954), also known as A. Turing, was a British mathematician and logician."

Output:
[
  {"subject": "Alan Turing", "predicate": "born", "object": "23 June 1912"},
  {"subject": "Alan Turing", "predicate": "died", "object": "7 June 1954"},
  {"subject": "Alan Turing", "predicate": "also known as", "object": "A. Turing"},
  {"subject": "Alan Turing", "predicate": "was", "object": "British"},
  {"subject": "Alan Turing", "predicate": "was", "object": "mathematician"},
  {"subject": "Alan Turing", "predicate": "was", "object": "logician"}
]

The user's message contains the input text.

Extract all atomic facts from that text following the rules above.
"""