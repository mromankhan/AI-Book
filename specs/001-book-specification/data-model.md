# Data Model: Physical AI: Humanoid & Robotics Systems Book

## Core Entities

### Book
- **Fields**: title, description, author, creationDate, lastUpdated
- **Relationships**: Contains many Parts
- **Validation**: Title is required, must have at least one Part
- **State**: Published/Unpublished (determines visibility)

### Part
- **Fields**: id, partNumber, title, description, order
- **Relationships**: Belongs to one Book, Contains many Chapters
- **Validation**: Must have a unique partNumber within the Book, title is required
- **State**: Draft/In Review/Published

### Chapter
- **Fields**: id, chapterNumber, title, content, estimatedReadingTime, partId
- **Relationships**: Belongs to one Part, Contains many Exercises
- **Validation**: Title is required, content must follow 6-part structure
- **State**: Draft/In Review/Published

### Chapter Structure (Required Format)
- **Overview Section**
  - Fields: learningObjectives, importance
  - Validation: Learning objectives must be clearly stated

- **Core Concepts Section**
  - Fields: concepts, diagrams, explanations
  - Validation: Must include diagrams-first approach, intuition-heavy explanations

- **Hands-on Section**
  - Fields: instructions, codeExamples, expectedOutput, toolsRequired
  - Validation: Must include step-by-step instructions, clear expected output
  - State: Simulation-compatible, Hardware-optional

- **Real-World Mapping Section**
  - Fields: realApplications, limitations, bridgingContent
  - Validation: Must explain how concepts apply to real robots

- **Exercises Section**
  - Fields: beginnerTasks, stretchChallenges
  - Validation: Must include at least one beginner task and one stretch challenge

- **Summary Section**
  - Fields: keyTakeaways, nextSteps
  - Validation: Must summarize key concepts and indicate what's next

### Exercise
- **Fields**: id, title, description, difficulty, chapterId
- **Relationships**: Belongs to one Chapter
- **Validation**: Difficulty must be Beginner, Intermediate, or Advanced
- **State**: Active/Archived

### User
- **Fields**: id, name, experienceLevel, progress, registrationDate
- **Relationships**: May have progress on many Chapters
- **Validation**: Experience level must be Beginner, Intermediate, or Advanced
- **State**: Active/Inactive

## Relationships Summary

```
Book (1) -----> (Many) Part (1) -----> (Many) Chapter (1) -----> (Many) Exercise
```

## Content Validation Rules

1. **Structure Validation**: Each Chapter must contain all 6 required sections
2. **Prerequisite Validation**: Content in later Parts/Chapters must only reference concepts already introduced
3. **Link Validation**: All internal navigation links must be valid
4. **Media Validation**: All diagrams and media must render properly
5. **Accessibility Validation**: Content must meet WCAG 2.1 AA standards

## State Transitions

### Chapter State Transitions
```
Draft -> In Review -> Published
Draft -> In Review -> Draft (with changes)
Published -> In Review (for updates)
```

### Book State Transitions
```
Unpublished -> Published (when all Parts are published)
Published -> In Review (when major updates are planned)