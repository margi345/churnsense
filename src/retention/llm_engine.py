import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv
from src.data.loader import load_config
from src.retention.prompts import (
    RETENTION_SYSTEM_PROMPT,
    RETENTION_USER_TEMPLATE,
    FEW_SHOT_EXAMPLE
)

load_dotenv()


# ── Retention Engine Class ─────────────────────────────────────────────────────
class RetentionEngine:
    """
    Uses an LLM to generate personalized retention strategies
    for at-risk customers based on their profile and SHAP risk factors.
    """

    def __init__(self):
        self.config = load_config()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model  = self.config["llm"]["model"]

    # ── Generate retention plan for one customer ───────────────────────────────
    def generate(
        self,
        customer_profile: dict,
        risk_factors: list,
        max_retries: int = None
    ) -> dict:
        """
        Args:
            customer_profile: dict with customer fields
            risk_factors: list of dicts from explain_local()
            max_retries: number of retries on failure

        Returns:
            dict with segment, urgency, summary, and 3 actions
        """
        if max_retries is None:
            max_retries = self.config["llm"]["max_retries"]

        # Format risk factors for prompt
        risk_factors_str = "\n".join([
            f"- {f['feature']}: {f['direction']} (magnitude: {f['magnitude']})"
            for f in risk_factors[:5]
        ])

        # Build user message
        user_message = RETENTION_USER_TEMPLATE.format(
            tenure          = customer_profile.get("tenure", "unknown"),
            monthly_charges = customer_profile.get("MonthlyCharges", "unknown"),
            contract_type   = customer_profile.get("Contract", "unknown"),
            internet_service = customer_profile.get("InternetService", "unknown"),
            total_services  = customer_profile.get("total_services", "unknown"),
            payment_method  = customer_profile.get("PaymentMethod", "unknown"),
            churn_prob      = round(customer_profile.get("churn_score", 0) * 100, 1),
            risk_tier       = customer_profile.get("risk_tier", "unknown"),
            segment         = customer_profile.get("segment_name", "unknown"),
            risk_factors    = risk_factors_str
        )

        # Add few-shot example to system prompt
        system_with_example = RETENTION_SYSTEM_PROMPT + "\n" + FEW_SHOT_EXAMPLE

        # ── Call LLM with retries ──────────────────────────────────────────────
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model       = self.model,
                    temperature = self.config["llm"]["temperature"],
                    max_tokens  = self.config["llm"]["max_tokens"],
                    messages    = [
                        {"role": "system", "content": system_with_example},
                        {"role": "user",   "content": user_message}
                    ]
                )

                raw_text = response.choices[0].message.content.strip()

                # Parse JSON response
                plan = json.loads(raw_text)

                # Validate structure
                assert "actions" in plan, "Missing actions key"
                assert len(plan["actions"]) >= 1, "No actions generated"

                return plan

            except json.JSONDecodeError as e:
                print(f"  JSON parse error (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)

            except Exception as e:
                print(f"  LLM error (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)

        # ── Fallback if LLM fails ──────────────────────────────────────────────
        return self._fallback_plan(customer_profile, risk_factors)

    # ── Rule-based fallback if LLM is unavailable ──────────────────────────────
    def _fallback_plan(self, customer_profile: dict, risk_factors: list) -> dict:
        """
        Returns a rule-based retention plan when LLM is unavailable.
        Ensures the API always returns something useful.
        """
        risk_tier = customer_profile.get("risk_tier", "high")
        contract  = customer_profile.get("Contract", "Month-to-month")
        charges   = customer_profile.get("MonthlyCharges", 70)

        actions = []

        # Rule 1: Contract upgrade offer
        if contract == "Month-to-month":
            actions.append({
                "type":            "discount",
                "title":           "Annual plan discount",
                "message":         "Switch to an annual plan and save 15% every month.",
                "offer":           "15% discount on annual contract",
                "expected_impact": "high"
            })

        # Rule 2: High charges → loyalty reward
        if float(charges) > 70:
            actions.append({
                "type":            "loyalty",
                "title":           "Loyalty reward for valued customer",
                "message":         "As one of our valued customers, you qualify for a special loyalty reward.",
                "offer":           "One month free + priority support",
                "expected_impact": "medium"
            })

        # Rule 3: Proactive support outreach
        actions.append({
            "type":            "outreach",
            "title":           "Personal account review",
            "message":         "We'd like to schedule a quick call to make sure you're getting the most from your plan.",
            "offer":           "Free account review + personalized plan recommendation",
            "expected_impact": "medium"
        })

        return {
            "segment":  customer_profile.get("segment_name", "unknown"),
            "urgency":  risk_tier,
            "summary":  "Customer flagged as at-risk based on behavioral signals.",
            "actions":  actions[:3],
            "source":   "fallback"
        }


# ── Run directly to test ───────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = RetentionEngine()

    test_profile = {
        "tenure":          2,
        "MonthlyCharges":  85.0,
        "Contract":        "Month-to-month",
        "InternetService": "Fiber optic",
        "PaymentMethod":   "Electronic check",
        "total_services":  2,
        "churn_score":     0.87,
        "risk_tier":       "critical",
        "segment_name":    "early-lifecycle"
    }

    test_risk_factors = [
        {"feature": "is_month_to_month", "direction": "increases churn risk", "magnitude": 0.44},
        {"feature": "is_new_customer",   "direction": "increases churn risk", "magnitude": 0.32},
        {"feature": "churn_risk_score",  "direction": "increases churn risk", "magnitude": 0.24},
        {"feature": "has_fiber",         "direction": "increases churn risk", "magnitude": 0.19},
        {"feature": "is_auto_payment",   "direction": "increases churn risk", "magnitude": 0.15},
    ]

    print("Generating retention plan...")
    plan = engine.generate(test_profile, test_risk_factors)

    print("\nRetention Plan:")
    print(f"  Segment:  {plan.get('segment')}")
    print(f"  Urgency:  {plan.get('urgency')}")
    print(f"  Summary:  {plan.get('summary')}")
    print(f"\n  Actions:")
    for i, action in enumerate(plan.get("actions", []), 1):
        print(f"\n  {i}. {action.get('title')}")
        print(f"     Type:    {action.get('type')}")
        print(f"     Message: {action.get('message')}")
        print(f"     Offer:   {action.get('offer')}")
        print(f"     Impact:  {action.get('expected_impact')}")