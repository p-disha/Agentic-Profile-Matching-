# Test Scenarios

## Scenario 1: Standard JD to Top Candidates Pipeline
**User Input**: "We are looking for a Frontend Developer with strong React experience. 3+ years experience required."
**Expected Agent Behavior**:
1. Parses the intent as a JD.
2. Extracts must-have ("React", "3+ years experience").
3. Searches mock candidate database.
4. Ranks candidates and outputs a report showing Jane Smith and John Doe.

## Scenario 2: Iterative Refinement
**User Input**: "Actually, let's also require TypeScript experience."
**Expected Agent Behavior**:
1. Parses intent as FEEDBACK.
2. Agent acknowledges feedback and routes back to extraction/search.
3. Ranks candidates again, prioritizing those with TypeScript (Jane Smith ranks higher).
4. Outputs the updated candidate report.

## Scenario 3: Head-to-Head Comparison Query
**User Input**: "Compare the top 2 matches side by side."
**Expected Agent Behavior**:
1. Parses intent as COMPARE.
2. Invokes the `compare_candidates` tool with the top 2 candidate IDs from state.
3. Outputs a head-to-head comparison markdown report.

## Scenario 4: Explainability
**User Input**: "Why did Jane rank higher than John?"
**Expected Agent Behavior**:
1. Parses intent as EXPLAIN.
2. LLM receives current shortlist and user query.
3. Outputs reasoning (e.g., "Jane has 4 years of experience and TypeScript, which match your refined criteria better than John's 3 years.").

## Scenario 5: Multi-Round Screening Deep Dive
**User Input**: "Generate interview questions for the top candidate."
**Expected Agent Behavior**:
*(Note: To run this fully, user can invoke it as FEEDBACK, but currently we mapped FEEDBACK to re-search. In a full system, you would have a specific tool for this. We will assume the LLM answers based on state or we add a 'generate questions' intent).*
1. Parses intent.
2. Uses the `generate_interview_questions` tool (or LLM knowledge).
3. Outputs targeted screening questions.
