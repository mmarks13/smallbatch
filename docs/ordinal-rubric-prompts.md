# Creating an Ordinal Rubric Prompt for a Small Model

Guidance for writing the `prompt` of a Smallbatch function whose output is an
ordinal integer scale. Small models follow ordered, observable thresholds far
more reliably than they follow holistic judgment, and a prompt written this way
raises teacher self-consistency, which bounds every candidate's achievable
fidelity.

## Define one construct

State exactly what the model is scoring in one sentence.

> Score the input using the ordinal scale based on [single construct].

Specify what evidence the model may consider and whether missing information
counts against the score.

## Use cumulative levels

Treat the scale as a sequence of ordered thresholds, not a set of independent
categories.

Each level should:

- include the requirements of the levels below it;
- add one meaningful, observable condition;
- represent a clear increase in the construct being measured.

## Define observable boundaries

Replace vague terms such as *poor*, *good*, and *excellent* with evidence that
can be identified in the input.

For every pair of adjacent levels, define:

- what must be present to move above the lower level;
- what additional evidence the higher level requires;
- what prevents an item from receiving the higher level.

Use this tie-breaking rule:

> Return the higher level only when all of its requirements are clearly
> supported. Otherwise, return the lower level.

## Keep uncertainty separate

Do not use a middle level to represent uncertainty.

Tell the model:

- Use only evidence contained in the input.
- Do not infer missing facts.
- When evidence for a higher level is absent or ambiguous, return the highest
  lower level that is fully supported.

Handle genuinely unscorable inputs separately from the ordinal scale.

## Include boundary examples

Use a small number of examples focused on commonly confused adjacent levels.
Prioritize examples near important decision boundaries rather than obvious
extremes.

## Require a constrained output

Ask the model to return only one valid value from the scale.

> Return exactly one of: [valid scale values].

## Prompt template

**TASK**

Score the INPUT using the ordinal scale below based only on [construct].
Return the highest level whose complete definition is supported.

**SCALE**

- **[Lowest level] — [anchor]:** [observable minimum or failure conditions]
- **[Next level] — [anchor]:** Meets the previous level and additionally
  [observable conditions]
- Continue this pattern for each level.
- **[Highest level] — [anchor]:** Meets the previous level and additionally
  [strongest observable condition]

**DECISION RULES**

1. Use only evidence contained in the INPUT.
2. Do not infer facts that are not stated.
3. Evaluate the levels in order from lowest to highest.
4. Return the highest level whose full requirements are satisfied.
5. If a higher level is only partially satisfied, return the lower level.
6. Do not use the score to express confidence or uncertainty.

**EXAMPLES**

Include a few representative examples emphasizing difficult adjacent-level
boundaries.

**INPUT**

{{input}}

**OUTPUT**

Return exactly one of: [valid scale values].

## Refining against measured disagreement

Smallbatch's calibration double-labels a sample so you can see where the
teacher contradicts itself. Refine by *sharpening* the boundary that produced
each disagreement — tighten a threshold, move a condition to the level it
belongs to, or delete an overlapping clause. Do not simply append more rules;
a longer rubric with unchanged boundaries usually lowers consistency.
