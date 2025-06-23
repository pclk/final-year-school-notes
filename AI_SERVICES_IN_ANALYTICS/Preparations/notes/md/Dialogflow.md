# Converting Follow-up Intents to Context-Linked Intents in Dialogflow

## Overview
This guide demonstrates how to recreate follow-up intent behavior using standard intents linked through contexts. The example uses an appointment scheduling flow with hair styling services.

## Original Flow (with Follow-up Intents)
```
Appt
└── Appt - yes
    ├── Color - yes
    └── Color - no
└── Appt - no
```

## Converted Flow (with Context-Linked Intents)

### 1. Main Appointment Intent
```yaml
Name: Appt
Training Phrases:
  - Book appointment
  - Schedule haircut
  - Make appointment
Output Context: 
  - appt-followup (lifespan: 2)
Parameters:
  - date (@sys.date)
  - time (@sys.time)
Response: 
  "Sure, book you for haircut on $date.original at $time.original ok?"
```

### 2. Appointment Confirmation Intent
```yaml
Name: Appt_Yes
Training Phrases:
  - yes
  - sure
  - okay
Input Context: 
  - appt-followup
Output Context: 
  - appt-yes-followup (lifespan: 2)
Parameters:
  - date: #appt-followup.date
  - time: #appt-followup.time
Response: 
  "Would you like to color your hair?"
```

### 3. Color Service Confirmation Intent
```yaml
Name: Color_Yes
Training Phrases:
  - yes
  - I want color
  - sure
Input Context: 
  - appt-yes-followup
Parameters:
  - date: #appt-yes-followup.date
  - time: #appt-yes-followup.time
Response: 
  "See you for haircut and color on $date.original at $time.original"
```

### 4. Color Service Rejection Intent
```yaml
Name: Color_No
Training Phrases:
  - no
  - just haircut
  - no color
Input Context: 
  - appt-yes-followup
Parameters:
  - date: #appt-yes-followup.date
  - time: #appt-yes-followup.time
Response: 
  "See you for haircut only on $date.original at $time.original"
```

### 5. Appointment Rejection Intent
```yaml
Name: Appt_No
Training Phrases:
  - no
  - cancel
  - not now
Input Context: 
  - appt-followup
Response: 
  "No worries. Let us know if you change your mind."
```

## Key Points
1. **Context Management**
   - Use output contexts to pass data to the next intent
   - Set appropriate context lifespans (usually 2)
   - Pass parameters through contexts when needed

2. **Parameter Access**
   - In the same intent: Use `$parameter.original`
   - From previous context: Use `#context-name.parameter`

3. **Context Chain**
   - Parameters must be explicitly passed through each context in the chain
   - Each intent should have matching input/output contexts

## Example Conversation Flow
```
User: "Book appointment"
Bot: "Sure, book you for haircut on [date] at [time] ok?"
User: "Yes"
Bot: "Would you like to color your hair?"
User: "Yes"
Bot: "See you for haircut and color on [date] at [time]"
```

## Common Issues
- Remember to pass parameters through contexts in each step
- Ensure context lifespans are sufficient for the conversation flow
- Use correct syntax for accessing parameters (`$` vs `#`)
- Match input/output context names exactly

This structure maintains the same functionality as follow-up intents while providing more flexibility in conversation flow management.
