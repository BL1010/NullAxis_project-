import json
import random

# Base templates and phrases for diversity in each class

tech_support_phrases = [
    "The app doesn't let me log in anymore",
    "Why is my account being logged out every hour?",
    "Getting an authentication error on mobile",
    "My dashboard won't load after the update",
    "I keep getting timeout errors when accessing the platform",
    "Photos are not syncing between devices",
    "The app crashes when I try to share a file",
    "The search bar returns irrelevant results",
    "I'm unable to reset my two-factor authentication",
    "Why can't I view the analytics section anymore?",
    "The software is not syncing my data",
    "I'm getting an error when uploading files",
    "The app is freezing on startup",
    "My subscription won't renew",
    "I found a bug in the latest update",
    "The website shows a 404 error",
    "I'm unable to download the latest version",
    "Error 500 appears when I upload files",
    "The app drains my battery quickly",
    "I’m having trouble with push notifications",
    "App crashes when opening notifications",
    "I can't connect to the server",
    "The app logs me out automatically",
    "The video calls are not connecting",
    "I'm experiencing delays in message delivery",
    "The app doesn’t save my preferences",
    "How do I clear the cache?",
    "I’m not receiving password reset emails",
    "The mobile app doesn't open",
    "The app freezes when playing videos",
    "I can’t sync my calendar",
    "How do I contact support?"
]

feature_request_phrases = [
    "I'd love to see a kanban board layout added",
    "Please include a timer for focus sessions",
    "Can you enable syncing with Apple Calendar?",
    "It would be helpful to have report auto-generation",
    "I’d like the ability to assign tasks to multiple people",
    "Can you add markdown support in notes?",
    "Request to add multi-device clipboard syncing",
    "I want to tag and categorize notifications",
    "Add a toggle for low-data usage mode",
    "Would be great to set recurring reminders",
    "Can you add dark mode?",
    "Please add multi-language support",
    "I would like a feature to export reports",
    "Add the ability to schedule notifications",
    "It would be great to have a calendar view",
    "Can you include voice commands?",
    "Please add integration with Slack",
    "I want to suggest adding dark theme",
    "Can the app support biometric login?",
    "Please add a way to archive completed tasks",
    "Feature request: improve search functionality",
    "Add support for exporting data to CSV",
    "Please provide more customization options",
    "Can you improve app startup speed?",
    "I want a feature for offline mode",
    "Add a dark mode toggle in settings",
    "Would love integration with Google Calendar",
    "Can you add customizable notifications?",
    "Please support exporting to PDF",
    "Add more font options for readability",
    "I want to be able to undo actions",
    "Please add a widget for quick access",
    "Can you add video tutorials inside the app?",
    "Please support multiple user profiles",
    "Feature request: dark mode scheduling",
    "Add the ability to pin important tasks",
    "Would be nice to have gesture controls",
    "Please add voice message support",
    "Can you add a feature to mark favorites?",
    "Add shortcut keys for power users",
    "Please support themes and skins",
    "Can you add a night reading mode?",
    "Would love integration with Dropbox",
    "Please add a progress tracker for tasks",
    "Feature request: automatic backups",
    "Can you support dark mode on mobile?",
    "Add the ability to customize the dashboard",
    "Please provide a feature for data encryption",
    "Would love an AI assistant integration",
    "Add social media sharing options",
    "Please add offline editing capabilities",
    "Can you add a feature to import contacts?",
    "Add integration with Microsoft Teams",
    "Would be great to have customizable dashboards"
]

sales_lead_phrases = [
    "We need 100 licenses for our new employees",
    "How do you handle onboarding for enterprise clients?",
    "Interested in a long-term contract — any discounts?",
    "Please send us the terms of your annual plan",
    "We want a pilot project before full purchase",
    "Our CTO would like to speak with your sales team",
    "Does your product support government compliance?",
    "Looking to bundle this with another service — is that possible?",
    "Can you create a custom onboarding package?",
    "We’re budgeting for Q3 — need a quote soon",
    "We are a team of 10 interested in your product",
    "I want to learn more about pricing",
    "Can you provide a demo for our company?",
    "Looking to purchase licenses for our organization",
    "We have 50 employees and want to buy",
    "Interested in enterprise solutions",
    "How much does your product cost for teams?",
    "I represent a startup and want pricing details",
    "Can you send me your sales brochure?",
    "We want to schedule a call with sales",
    "Our organization is interested in bulk licensing",
    "Can you help us with onboarding?",
    "I want to know about your product's pricing plans",
    "Looking to negotiate a deal for our company",
    "How can I get a volume discount?",
    "What is the pricing for enterprise customers?",
    "Can you provide a custom quote for our team?",
    "We want to integrate this product into our workflow",
    "I want to discuss partnership opportunities",
    "Do you offer discounts for non-profits?",
    "Please send me more information about your plans",
    "Our company is interested in annual subscriptions",
    "How do I become a reseller?",
    "Is there a free trial for enterprise clients?",
    "We have 200 employees and want pricing info",
    "Can we get a demo next week?",
    "How do you handle bulk orders?",
    "I want to learn about your service-level agreements",
    "Are there special plans for educational institutions?",
    "Can you provide training for our staff?",
    "What support options are included with the purchase?",
    "I’m interested in a quote for our department",
    "We want to upgrade to a premium plan",
    "Can you help with migration to your platform?",
    "I want to know about volume licensing",
    "Our team is growing; what options do you offer?",
    "Can you assist with implementation?",
    "Is there a dedicated account manager?",
    "How long does the onboarding process take?",
    "Are there discounts for long-term contracts?",
    "I want to schedule a sales call",
    "Can I get a demo for our executives?",
    "What is your pricing per user?",
    "Can you customize the product for our needs?",
    "We want integration with our CRM",
    "Please provide case studies for your product",
    "Are there partner programs available?",
    "What are your payment terms?",
    "Is there a discount for early payment?",
    "Can you provide a roadmap of upcoming features?",
    "We want to pilot the software before buying"
]

def generate_examples(base_phrases, label, count):
    examples = []
    for _ in range(count):
        base = random.choice(base_phrases)
        # No prefix/suffix variations, use base text directly
        text = base
        examples.append({"text": text, "label": label})
    return examples

# Generate 500 examples each
random.seed(42)  # For reproducibility

tech_support_examples = generate_examples(tech_support_phrases, "Technical Support", 500)
feature_request_examples = generate_examples(feature_request_phrases, "Product Feature Request", 500)
sales_lead_examples = generate_examples(sales_lead_phrases, "Sales Lead", 500)

# Combine all
all_examples = tech_support_examples + feature_request_examples + sales_lead_examples

# Shuffle the combined dataset to mix examples
random.shuffle(all_examples)

# Save as JSONL file
output_path = "customer_intent_dataset.jsonl"
with open(output_path, "w", encoding="utf-8") as f:
    for example in all_examples:
        f.write(json.dumps(example) + "\n")

print(f"Dataset saved to {output_path}")
