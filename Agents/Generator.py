import os
from typing import Dict, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


class LinkedInPostAgent:
    """Interactive agent for generating LinkedIn posts"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with Gemini model"""
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.9,
            max_output_tokens=2048
        )
        
        self.prompt_templates = self._load_prompt_templates()
        self.audiences = self._load_audiences()
    
    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load predefined prompt templates"""
        return {
            "1": {
                "name": "Personal Story",
                "prompt": """Write a LinkedIn post sharing a personal story about {topic}.
                Start with a relatable moment or challenge you faced.
                Build the narrative with specific details and emotions.
                Share what you learned and how you grew.
                End with a takeaway that others can apply.

                Make it authentic, vulnerable, and inspiring."""
            },
            
            "2": {
                "name": "Quick Tips List",
                "prompt": """Write a LinkedIn post with actionable tips about {topic}.
                
                Start with a bold hook that promises value.
                List 3-5 specific, actionable tips.
                For each tip, add a brief explanation or example.
                Use emojis for visual breaks (optional).
                End with a question to spark discussion.

                Keep it scannable and practical."""
            },
            
            "3": {
                "name": "Controversial Opinion",
                "prompt": """Write a LinkedIn post challenging common thinking about {topic}.
                
                Open with an unpopular opinion or hot take.
                Explain why the conventional wisdom is wrong.
                Support your view with examples or data.
                Acknowledge the opposing perspective respectfully.
                End with a thought-provoking question.

                Be bold but professional."""
            },
            
            "4": {
                "name": "Behind-the-Scenes",
                "prompt": """Write a LinkedIn post revealing behind-the-scenes insights about {topic}.
                
                Share something most people don't know or see.
                Include specific numbers, processes, or details.
                Explain why this matters to your audience.
                Add transparency and authenticity.
                End with an invitation to share their experiences.

                Make it feel like insider knowledge."""
            },
            
            "5": {
                "name": "Trend Analysis",
                "prompt": """Write a LinkedIn post analyzing a trend related to {topic}.
                
                Identify an emerging or current trend.
                Explain what's driving this trend.
                Share your perspective on its impact.
                Predict where it's heading.
                End with how your audience should respond.

                Be insightful and forward-thinking."""
            },
            
            "6": {
                "name": "Motivational Message",
                "prompt": """Write an inspiring LinkedIn post about {topic}.
                
                Open with a powerful statement or quote.
                Share a relevant struggle or setback (yours or someone's).
                Highlight the breakthrough or transformation.
                Connect it to universal human experiences.
                End with an uplifting call to action.

                Make it emotionally resonant and empowering."""
            }
        }
    
    def _load_audiences(self) -> Dict[str, str]:
        """Load audience profiles"""
        return {
            "1": {"name": "Startup Founders", "desc": "Entrepreneurs building tech startups, interested in growth, funding, and scaling"},
            "2": {"name": "Marketing Professionals", "desc": "Marketers focused on strategy, campaigns, and brand growth"},
            "3": {"name": "Software Developers", "desc": "Engineers and developers interested in tech, coding, and career growth"},
            "4": {"name": "Sales Professionals", "desc": "Sales people focused on prospecting, closing deals, and relationship building"},
            "5": {"name": "Job Seekers", "desc": "Professionals seeking career opportunities and personal branding advice"},
            "6": {"name": "Business Leaders", "desc": "Executives and managers interested in leadership and strategy"},
            "7": {"name": "Freelancers", "desc": "Independent professionals building their personal brand and client base"},
            "8": {"name": "General Professional", "desc": "Broad professional audience across industries"}
        }
    
    def _humanization_rules(self) -> str:
        """Rules for making content sound human"""
        return """
            HUMANIZATION RULES (CRITICAL):
            - Write like you're talking to a friend over coffee
            - Use "I," "you," "we" - make it personal
            - Short paragraphs (2-3 lines max)
            - Add line breaks for breathing room
            - Use simple, everyday words
            - Show emotion and personality
            - Admit mistakes or challenges
            - Ask questions to create dialogue
            - Vary sentence length naturally

            DO NOT:
            - Use corporate buzzwords (leverage, synergy, paradigm)
            - Use robotic or formal language
            - Write long walls of text
            - Use generic motivational quotes
            - Use "I hope this helps" or AI-isms
        """
    
    def show_welcome(self):
        """Display welcome message"""
        print("\n" + "="*60)
        print("LINKEDIN POST GENERATOR")
        print("   Powered by LangChain + Google Gemini")
        print("="*60 + "\n")
    
    def select_audience(self) -> str:
        """Interactive audience selection"""
        print("Who is your TARGET AUDIENCE?\n")
        
        for key, value in self.audiences.items():
            print(f"  {key}. {value['name']}")
        
        choice = input("\nEnter number (or type custom audience): ").strip()
        
        if choice in self.audiences:
            selected = self.audiences[choice]['desc']
            print(f"Selected: {self.audiences[choice]['name']}\n")
            return selected
        else:
            print(f"Custom audience: {choice}\n")
            return f"Professionals interested in {choice}"
    
    def choose_template_or_custom(self) -> tuple[bool, Optional[str], Optional[str]]:
        """Ask user: template or custom prompt?"""
        print("="*60)
        print("\nPROMPT SELECTION\n")
        
        use_template = input("Do you want to use a PROMPT TEMPLATE? (yes/no): ").strip().lower()
        
        if use_template in ['yes', 'y']:
            return True, self.select_template(), None
        else:
            return False, None, self.get_custom_prompt()
    
    def select_template(self) -> str:
        """Interactive template selection"""
        print("\nAvailable Prompt Templates:\n")
        
        for key, value in self.prompt_templates.items():
            print(f"  {key}. {value['name']}")
        
        choice = input("\nSelect template number: ").strip()
        
        if choice in self.prompt_templates:
            template = self.prompt_templates[choice]
            print(f"Selected: {template['name']}\n")
            return template['prompt']
        else:
            print("Invalid choice. Using Personal Story template.\n")
            return self.prompt_templates['1']['prompt']
    
    def get_custom_prompt(self) -> str:
        """Get custom prompt from user"""
        print("\nEnter your CUSTOM PROMPT:")
        print("   (Describe exactly what kind of post you want)\n")
        
        lines = []
        print("   Type your prompt (press Enter twice when done):\n")
        
        while True:
            line = input()
            if line == "" and lines and lines[-1] == "":
                break
            lines.append(line)
        
        custom_prompt = "\n".join(lines[:-1])  # Remove last empty line
        print("\nCustom prompt received!\n")
        return custom_prompt
    
    def get_topic(self) -> str:
        """Get the main topic from user"""
        print("="*60)
        print("\nWhat's your POST TOPIC?")
        print("   (Be specific about what you want to discuss)\n")
        
        topic = input("Topic: ").strip()
        print()
        return topic
    
    def generate_post(self, prompt: str, topic: str, audience: str) -> str:
        """Generate the LinkedIn post"""
        print("Generating your LinkedIn post...")
        print("   (This may take 10-20 seconds)\n")
        
        # Create the final prompt
        final_prompt = PromptTemplate(
            input_variables=["prompt", "topic", "audience", "rules"],
            template="""
                You are an expert LinkedIn content creator known for viral, engaging posts.

                USER'S PROMPT/INSTRUCTIONS:
                {prompt}

                TOPIC: {topic}

                TARGET AUDIENCE: {audience}

                {rules}

                IMPORTANT FORMATTING:
                - Start with a strong hook (first 1-2 lines)
                - Use line breaks every 2-3 sentences
                - Add emojis sparingly if they fit naturally
                - End with a clear call-to-action (question, invitation to comment, etc.)
                - Include 3-5 relevant hashtags at the end

                Generate a complete LinkedIn post that follows these guidelines and sounds authentically human.
            """
        )
        
        # Create chain using LCEL (LangChain Expression Language)
        chain = final_prompt | self.llm
        
        result = chain.invoke({
            "prompt": prompt,
            "topic": topic,
            "audience": audience,
            "rules": self._humanization_rules()
        })
        
        # Extract content from AIMessage if needed
        if hasattr(result, 'content'):
            result = result.content
        
        return result.strip()
    
    def display_post(self, post: str):
        """Display the generated post"""
        print("\n" + "="*60)
        print("YOUR LINKEDIN POST")
        print("="*60 + "\n")
        print(post)
        print("\n" + "="*60 + "\n")
    
    def parse_post_sections(self, post: str) -> Dict[str, str]:
        """Parse post into sections for editing"""
        sections = {
            "hook": "",
            "body": "",
            "cta": "",
            "hashtags": "",
            "full": post
        }
        
        lines = post.split('\n')
        current_section = "body"
        hook_found = False
        
        # Try to identify sections
        for i, line in enumerate(lines):
            line_lower = line.strip().lower()
            
            # Detect hook (usually first 1-2 non-empty lines)
            if not hook_found and line.strip() and i < 3:
                sections["hook"] += line + "\n"
                hook_found = True
                continue
            
            # Detect hashtags (lines starting with #)
            if line.strip().startswith('#'):
                sections["hashtags"] += line + "\n"
                continue
            
            # Detect CTA (questions, "What do you think?", etc.)
            if any(phrase in line_lower for phrase in ['what do you think', 'comment below', 'share your thoughts', '?']):
                if not sections["cta"]:
                    sections["cta"] = line + "\n"
                else:
                    sections["cta"] += line + "\n"
                continue
            
            # Everything else is body
            if line.strip():
                sections["body"] += line + "\n"
        
        # Clean up sections
        for key in sections:
            sections[key] = sections[key].strip()
        
        return sections
    
    def edit_post(self, post: str) -> str:
        """Allow user to edit specific parts of the generated post"""
        sections = self.parse_post_sections(post)
        
        while True:
            print("\n" + "="*60)
            print("EDIT POST")
            print("="*60)
            print("\nWhich part would you like to edit?")
            print("1. Hook (Opening lines)")
            print("2. Body (Main content)")
            print("3. Call-to-Action (Ending)")
            print("4. Hashtags")
            print("5. View current post")
            print("6. Done editing")
            
            choice = input("\nYour choice (1-6): ").strip()
            
            if choice == "1":
                print("\nCurrent Hook:")
                print("-" * 40)
                print(sections["hook"] if sections["hook"] else "(Not detected)")
                print("-" * 40)
                print("\nEnter new hook (press Enter twice when done):")
                new_hook = []
                while True:
                    line = input()
                    if line == "" and new_hook and new_hook[-1] == "":
                        break
                    new_hook.append(line)
                sections["hook"] = "\n".join(new_hook[:-1]).strip()
                print("Hook updated!\n")
            
            elif choice == "2":
                print("\nCurrent Body:")
                print("-" * 40)
                print(sections["body"] if sections["body"] else "(Not detected)")
                print("-" * 40)
                print("\nEnter new body (press Enter twice when done):")
                new_body = []
                while True:
                    line = input()
                    if line == "" and new_body and new_body[-1] == "":
                        break
                    new_body.append(line)
                sections["body"] = "\n".join(new_body[:-1]).strip()
                print("Body updated!\n")
            
            elif choice == "3":
                print("\nCurrent Call-to-Action:")
                print("-" * 40)
                print(sections["cta"] if sections["cta"] else "(Not detected)")
                print("-" * 40)
                print("\nEnter new call-to-action (press Enter twice when done):")
                new_cta = []
                while True:
                    line = input()
                    if line == "" and new_cta and new_cta[-1] == "":
                        break
                    new_cta.append(line)
                sections["cta"] = "\n".join(new_cta[:-1]).strip()
                print("Call-to-Action updated!\n")
            
            elif choice == "4":
                print("\nCurrent Hashtags:")
                print("-" * 40)
                print(sections["hashtags"] if sections["hashtags"] else "(Not detected)")
                print("-" * 40)
                print("\nEnter new hashtags (press Enter twice when done):")
                new_hashtags = []
                while True:
                    line = input()
                    if line == "" and new_hashtags and new_hashtags[-1] == "":
                        break
                    new_hashtags.append(line)
                sections["hashtags"] = "\n".join(new_hashtags[:-1]).strip()
                print("Hashtags updated!\n")
            
            elif choice == "5":
                # Reconstruct and display the post
                edited_post = self.reconstruct_post(sections)
                print("\n" + "="*60)
                print("CURRENT POST")
                print("="*60 + "\n")
                print(edited_post)
                print("\n" + "="*60 + "\n")
            
            elif choice == "6":
                # Reconstruct the final post
                edited_post = self.reconstruct_post(sections)
                return edited_post
            
            else:
                print("Invalid choice. Please enter 1-6.\n")
    
    def reconstruct_post(self, sections: Dict[str, str]) -> str:
        """Reconstruct the post from edited sections"""
        parts = []
        
        if sections["hook"]:
            parts.append(sections["hook"])
        
        if sections["body"]:
            parts.append(sections["body"])
        
        if sections["cta"]:
            parts.append(sections["cta"])
        
        if sections["hashtags"]:
            parts.append(sections["hashtags"])
        
        # If no sections were parsed, return original
        if not any(sections[key] for key in ["hook", "body", "cta", "hashtags"]):
            return sections["full"]
        
        return "\n\n".join(parts)
    
    def run(self):
        """Main interactive flow"""
        self.show_welcome()
        
        # Step 1: Select audience
        audience = self.select_audience()
        
        # Step 2: Choose template or custom prompt
        use_template, template_prompt, custom_prompt = self.choose_template_or_custom()
        
        # Step 3: Get topic
        topic = self.get_topic()
        
        # Step 4: Generate post
        prompt = template_prompt if use_template else custom_prompt
        post = self.generate_post(prompt, topic, audience)
        
        # Step 5: Display result
        self.display_post(post)
        
        # Step 6: Ask if they want to edit the post
        edit_choice = input("Would you like to edit this post? (yes/no): ").strip().lower()
        if edit_choice in ['yes', 'y']:
            post = self.edit_post(post)
            self.display_post(post)
        
        # Step 7: Ask if they want to use another template
        again = input("Use another template? (yes/no): ").strip().lower()
        if again in ['yes', 'y']:
            print("\n")
            self.run()
        else:
            print("\nThanks for using LinkedIn Post Generator!\n")


def main():
    """Entry point"""
    # Get API key from environment file (.env)
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if api_key:
        agent = LinkedInPostAgent(api_key=api_key)
    else:
        agent = LinkedInPostAgent()
    
    # Run the interactive app
    agent.run()


if __name__ == "__main__":
    main()