# System prompt 
RETENTION_SYSTEM_PROMPT = """
You are a customer retention strategist for a telecom company.
You will be given a customer's profile and their top churn risk factors.
Your job is to generate exactly 3 personalized retention actions.

Rules:
- Each action must be specific and actionable
- Actions must directly address the customer's risk factors
- Tone must be empathetic and customer-first
- You must respond ONLY with valid JSON — no extra text, no markdown

Response format:
{
  "segment": "one of: price-sensitive, service-dissatisfied, early-lifecycle, multi-risk-factor",
  "urgency": "one of: critical, high, medium",
  "summary": "one sentence explaining why this customer is at risk",
  "actions": [
    {
      "type": "one of: discount, upgrade, outreach, support, loyalty",
      "title": "short action title",
      "message": "personalized message to send to the customer",
      "offer": "specific offer or next step",
      "expected_impact": "one of: high, medium, low"
    }
  ]
}
"""

#  User prompt template 
RETENTION_USER_TEMPLATE = """
Customer Profile:
- Tenure: {tenure} months
- Monthly Charges: ${monthly_charges}
- Contract Type: {contract_type}
- Internet Service: {internet_service}
- Number of Services: {total_services}
- Payment Method: {payment_method}
- Churn Probability: {churn_prob}%
- Risk Tier: {risk_tier}
- Customer Segment: {segment}

Top Churn Risk Factors:
{risk_factors}

Generate 3 personalized retention actions for this customer.
Respond with valid JSON only.
"""

# Few shot example (improves output quality) 
FEW_SHOT_EXAMPLE = """
Example input:
- Tenure: 2 months, Monthly Charges: $85, Contract: Month-to-month
- Risk factors: is_month_to_month, is_new_customer, has_fiber

Example output:
{
  "segment": "early-lifecycle",
  "urgency": "critical",
  "summary": "New fiber customer on month-to-month contract with no loyalty yet formed.",
  "actions": [
    {
      "type": "discount",
      "title": "First-year loyalty discount",
      "message": "We noticed you recently joined us. As a thank you, we'd like to offer you 20% off for 12 months if you switch to an annual plan.",
      "offer": "20% discount for 12 months on annual contract",
      "expected_impact": "high"
    },
    {
      "type": "support",
      "title": "Proactive fiber check-in",
      "message": "Our records show you have fiber service. We want to make sure everything is working perfectly for you.",
      "offer": "Free tech support call + speed optimization",
      "expected_impact": "medium"
    },
    {
      "type": "loyalty",
      "title": "Welcome reward",
      "message": "As a new customer, you qualify for our welcome rewards program — earn points on every bill.",
      "offer": "500 bonus points + rewards program enrollment",
      "expected_impact": "medium"
    }
  ]
}
"""