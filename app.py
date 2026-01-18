import os
import streamlit as st
from typing import Dict, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Try to import pyperclip, fallback if not available
try:
    import pyperclip
    HAS_PYPERCLIP = True
except ImportError:
    HAS_PYPERCLIP = False

load_dotenv()

# --- UI CONFIGURATION ---
st.set_page_config(
    page_title="LinkedIn Post AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern, clean CSS
st.markdown("""
    <style>
    /* Main container */
    .main {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
        padding: 0.5rem 1rem 2rem 1rem;
    }
    
    /* Reduce top padding from Streamlit */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 2rem;
    }
    
    /* Hide sidebar */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 0.5rem 0 1rem 0;
        margin-bottom: 1rem;
        margin-top: 0;
    }
    
    h1 {
        color: #0a66c2;
        font-weight: 800;
        font-size: 2.5rem;
        margin-top: 0;
        margin-bottom: 0.25rem;
        letter-spacing: -0.5px;
    }
    
    h2 {
        color: #1d1d1f;
        font-weight: 700;
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #1d1d1f;
        font-weight: 600;
        font-size: 1.2rem;
        margin-bottom: 0.75rem;
    }
    
    /* Card styling */
    .config-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    
    .config-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.08);
        transform: translateY(-2px);
    }
    
    /* Buttons */
    .stButton>button {
        border-radius: 8px;
        background-color: #0a66c2;
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
        font-size: 1rem;
    }
    
    .stButton>button:hover {
        background-color: #004182;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(10, 102, 194, 0.3);
    }
    
    /* Download button */
    .stDownloadButton>button {
        background-color: #057642;
        color: white;
        font-weight: 600;
        border-radius: 8px;
    }
    
    .stDownloadButton>button:hover {
        background-color: #004d2a;
    }
    
    /* Input fields */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus,
    .stTextArea>div>div>textarea:focus {
        border-color: #0a66c2;
        box-shadow: 0 0 0 3px rgba(10, 102, 194, 0.1);
        outline: none;
    }
    
    /* Selectbox */
    .stSelectbox>div>div>select {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
    }
    
    /* Radio buttons */
    .stRadio>div {
        gap: 1rem;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0a66c2;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #666;
        font-weight: 500;
    }
    
    /* Success/Info messages */
    .stSuccess {
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #057642;
    }
    
    .stWarning {
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #ffc107;
    }
    
    .stError {
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #dc3545;
    }
    
    .stInfo {
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #0a66c2;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 2px solid #e0e0e0;
    }
    
    /* Section badges */
    .section-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        background: linear-gradient(135deg, #0a66c2, #004182);
        color: white;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 1rem;
    }
    
    /* Preview card */
    .preview-container {
        background: white;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        max-width: 600px;
        margin: 0 auto;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        h1 {
            font-size: 2rem;
        }
        .config-card {
            padding: 1rem;
        }
    }
    
    /* Hide Streamlit menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

class LinkedInPostAgent:
    """Agent for generating LinkedIn posts"""

    def __init__(self, api_key: Optional[str] = None):
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0.7,  # Reduced for better control
            max_output_tokens=1500  # Reduced to prevent overly long outputs
        )

        self.prompt_templates = self._load_prompt_templates()
        self.audiences = self._load_audiences()
        self.default_tone = "Professional"
        self.default_length = "Medium"

    def _load_prompt_templates(self) -> Dict[str, str]:
        return {
            "Personal Story": (
                "Write a LinkedIn post sharing a personal or experiential story related to {topic}. "
                "Start with a relatable moment or challenge. "
                "Build the narrative with specific details. "
                "Highlight growth or learning. "
                "End with a takeaway or reflective question."
            ),
            "Quick Tips List": (
                "Write a LinkedIn post offering actionable tips about {topic}. "
                "Open with a bold hook. "
                "List 3–5 concise tips. "
                "End with a question."
            ),
            "Controversial Opinion": (
                "Challenge common beliefs about {topic}. "
                "Open with a respectful hot take. "
                "Explain your reasoning. "
                "End with a discussion-driving question."
            ),
            "Behind-the-Scenes": (
                "Reveal behind-the-scenes insights about {topic}. "
                "Explain processes and trade-offs. "
                "Share why it matters."
            ),
            "Trend Analysis": (
                "Analyze a trend related to {topic}. "
                "Explain what's changing, why it matters, and what to do next."
            ),
            "Motivational Message": (
                "Write an uplifting post about {topic}. "
                "Acknowledge a challenge. "
                "Shift to growth and encouragement."
            ),
            "Lesson Learned": (
                "Share a clear lesson learned about {topic}. "
                "Explain briefly how it was learned. "
                "End with a reflective question."
            ),
            "Myth vs Reality": (
                "Debunk a common myth about {topic}. "
                "Contrast myth vs reality clearly."
            ),
            "How-To / Framework": (
                "Explain how to approach {topic} using a simple framework."
            )
        }

    def _load_audiences(self) -> Dict[str, str]:
        return {
            "Startup Founders": "Entrepreneurs building or scaling startups",
            "Marketing Professionals": "Marketers focused on growth and branding",
            "Software Developers": "Engineers focused on tech and career growth",
            "AI / ML / GenAI Professionals": (
                "Professionals working with AI, ML, GenAI, and LLM systems"
            ),
            "Data Analysts & Data Scientists": (
                "Professionals focused on analytics, insights, and data-driven decisions"
            ),
            "Sales Professionals": "Sales professionals focused on revenue",
            "Job Seekers": "Professionals exploring new roles",
            "Business Leaders": "Managers and executives",
            "Freelancers": "Independent professionals building a brand",
            "Students & Early Career": "Students and early professionals",
            "Creators & Builders": "Content creators and builders",
            "General Professionals": "Broad professional audience"
        }

    def _humanization_rules(self) -> str:
        return """
AI WRITING RULES (TRANSPARENT & LINKEDIN-NATIVE):
- Write clearly and professionally.
- Do not pretend to be a specific human.
- Keep paragraphs short (1–2 sentences).
- Avoid buzzwords and clichés.
- No fake personal experiences.
- Focus on usefulness and clarity.
"""

    def generate_post(self, user_instructions: str, topic: str, audience: str, tone: str = "Professional", length: str = "Medium") -> str:
        tone_instructions = {
            "Professional": (
                "Formal, polished, and business-appropriate. "
                "Clear and confident language with a respectful, authoritative voice."
            ),
            "Casual": (
                "Relaxed and conversational. "
                "Friendly, natural, and approachable—like talking to a colleague over coffee."
            ),
            "Conversational": (
                "Personal and engaging. "
                "Uses simple language, short sentences, and direct questions to involve the reader."
            ),
            "Controversial": (
                "Bold and thought-provoking while remaining respectful. "
                "Challenges conventional wisdom and invites discussion without being offensive."
            ),
            "Empathetic": (
                "Warm, supportive, and emotionally intelligent. "
                "Acknowledges challenges, validates experiences, and builds human connection."
            ),
            "Educational": (
                "Informative and insight-driven. "
                "Explains concepts clearly, shares practical examples, and focuses on learning value."
            ),
            "Inspirational": (
                "Motivational and uplifting. "
                "Encourages growth, confidence, and action through positive messaging."
            ),
            "Storytelling": (
                "Narrative-driven and relatable. "
                "Uses real-life experiences, lessons, and reflections to deliver a message."
            ),
            "Direct": (
                "Clear, concise, and to the point. "
                "Minimal fluff, strong statements, and actionable takeaways."
            )
        }

        length_instructions = {
            "Short": {
                "description": "Very concise and skimmable. 1–2 short paragraphs.",
                "word_count": "50–100 words MAXIMUM",
                "strict_limit": 100
            },
            "Medium": {
                "description": "Standard LinkedIn post. 3–5 short paragraphs with line breaks.",
                "word_count": "150–250 words MAXIMUM",
                "strict_limit": 250
            },
            "Long": {
                "description": "In-depth, value-driven. 6–10 short paragraphs with strong spacing.",
                "word_count": "300–500 words MAXIMUM",
                "strict_limit": 500
            }
        }

        length_config = length_instructions.get(length, length_instructions["Medium"])
        
        final_prompt = PromptTemplate(
            input_variables=["user_instructions", "topic", "audience", "rules", "tone_guide", "length_desc", "word_limit"],
            template="""
You are a seasoned LinkedIn content creator known for high-engagement, human-sounding posts.

CRITICAL: You MUST strictly follow the word count limit specified below. This is non-negotiable.

CONTEXT & GOAL:
Write a LinkedIn post that aligns with the user's intent and feels authentic, thoughtful, and platform-native.

USER INSTRUCTIONS:
{user_instructions}

POST DETAILS:
- Topic: {topic}
- Target Audience: {audience}
- Tone: {tone_guide}

LENGTH REQUIREMENT (STRICTLY ENFORCE):
{length_desc}
ABSOLUTE WORD LIMIT: {word_limit}
You MUST stay within this word count. Count your words as you write. Do NOT exceed this limit under any circumstances.

CONTENT RULES:
{rules}

STRUCTURE & STYLE GUIDELINES:
- Open with a strong hook in the first 1–2 lines (bold statement, question, or insight)
- Use short paragraphs (1–2 sentences max) with frequent line breaks
- Avoid emojis unless they naturally fit the selected tone
- Avoid generic phrases, clichés, and obvious AI patterns
- Use clear, simple language—write like a real LinkedIn creator, not a blog
- Be concise and punchy - every word must earn its place

ENGAGEMENT OPTIMIZATION:
- Share a clear insight, lesson, or takeaway
- Encourage interaction with a thoughtful question or call-to-action
- Do not over-sell or sound promotional

HASHTAGS & ENDING:
- End the post with 3–5 relevant, niche-specific hashtags
- Place hashtags on a new line at the very end
- Do not include hashtags within the main content

FINAL CHECK BEFORE SUBMITTING:
1. Count the total words (excluding hashtags)
2. Ensure you are UNDER the {word_limit} word limit
3. The post should feel human, credible, and experience-driven
4. Prioritize clarity, relatability, and skimmability
5. If you're over the limit, cut content aggressively - quality over quantity

Write the LinkedIn post now. Remember: STAY UNDER {word_limit} WORDS.
"""
        )

        chain = final_prompt | self.llm

        try:
            result = chain.invoke({
                "user_instructions": user_instructions.strip(),
                "topic": topic.strip(),
                "audience": audience.strip(),
                "rules": self._humanization_rules(),
                "tone_guide": tone_instructions.get(tone, tone_instructions["Professional"]),
                "length_desc": length_config["description"],
                "word_limit": length_config["strict_limit"]
            })

            # Handle different response types
            if hasattr(result, "content"):
                content = result.content
            elif isinstance(result, str):
                content = result
            else:
                content = str(result)
            
            content = content.strip()
            
            # Validate and trim if necessary
            words = content.split()
            word_count = len([w for w in words if not w.startswith('#')])
            
            if word_count > length_config["strict_limit"] * 1.2:  # If 20% over limit
                # Extract hashtags if present
                lines = content.split('\n')
                hashtag_line = None
                main_content_lines = []
                
                for line in lines:
                    if line.strip().startswith('#') or all(word.startswith('#') for word in line.strip().split() if word):
                        hashtag_line = line
                    else:
                        main_content_lines.append(line)
                
                main_content = '\n'.join(main_content_lines).strip()
                
                # Trim to word limit
                main_words = main_content.split()
                trimmed_words = main_words[:length_config["strict_limit"]]
                content = ' '.join(trimmed_words)
                
                # Re-add hashtags
                if hashtag_line:
                    content = f"{content}\n\n{hashtag_line}"
            
            return content
            
        except Exception as e:
            raise Exception(f"Error generating post: {str(e)}")

# --- MAIN APP FLOW ---

def main():
    # Initialize session state
    if 'post_history' not in st.session_state:
        st.session_state.post_history = []
    if 'current_post' not in st.session_state:
        st.session_state.current_post = ""
    if 'config_expanded' not in st.session_state:
        st.session_state.config_expanded = False
    if 'generation_triggered' not in st.session_state:
        st.session_state.generation_triggered = False
    
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        st.error("⚠️ API Key not found! Please set GOOGLE_API_KEY in your .env file.")
        st.info("💡 Create a .env file in your project root with: GOOGLE_API_KEY=your_api_key_here")
        st.stop()
    
    try:
        agent = LinkedInPostAgent(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        st.stop()

    # Header Section
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("LinkedIn Post Generator")
    st.markdown("**Generate high-engagement, human-sounding posts in seconds**")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Configuration Section (Collapsible)
    with st.expander("⚙️ Post Configuration", expanded=st.session_state.config_expanded):
        st.markdown('<div class="section-badge">Settings</div>', unsafe_allow_html=True)
        
        # Configuration in columns for better layout
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            st.markdown("**Target Audience**")
            audience_name = st.selectbox(
                "Who is your audience?", 
                list(agent.audiences.keys()),
                help="Select the primary audience for your post",
                label_visibility="collapsed"
            )
            audience_desc = agent.audiences[audience_name]
            st.caption(f"*{audience_desc}*")
            
            st.markdown("**Content Tone**")
            tone = st.radio(
                "Select tone:",
                ["Professional", "Casual", "Conversational", "Controversial", "Empathetic", 
                 "Educational", "Inspirational", "Storytelling", "Direct"],
                horizontal=False,
                label_visibility="collapsed"
            )
        
        with config_col2:
            st.markdown("**Post Length**")
            length = st.radio(
                "Select length:",
                ["Short", "Medium", "Long"],
                index=1,
                horizontal=True,
                label_visibility="collapsed"
            )
            
            st.markdown("**Content Style**")
            use_custom = st.checkbox("Use custom prompt", help="Check to write your own instructions")
        
        # Template Selection
        if not use_custom:
            st.markdown("**Choose Template**")
            template_name = st.selectbox(
                "Choose a template:", 
                list(agent.prompt_templates.keys()),
                help="Select a pre-built template style",
                label_visibility="collapsed"
            )
            selected_prompt = agent.prompt_templates[template_name]
            with st.expander("📋 Template Preview"):
                st.caption(selected_prompt)
        else:
            st.markdown("**Custom Instructions**")
            selected_prompt = st.text_area(
                "Custom instructions:", 
                placeholder="e.g., Write a funny, engaging post about coding that includes a personal anecdote...",
                height=100,
                label_visibility="collapsed"
            )
    
    st.markdown("---")
    
    # Topic Input and Generate Section
    st.markdown('<div class="section-badge">Create Post</div>', unsafe_allow_html=True)
    
    topic_col1, topic_col2 = st.columns([4, 1])
    with topic_col1:
        topic = st.text_input(
            "**What is your post about?**", 
            placeholder="e.g., The future of Remote Work in 2026",
            help="Enter the main topic or theme for your LinkedIn post",
            label_visibility="collapsed"
        )
    
    with topic_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        generate_btn = st.button("✨ Generate Post", type="primary", use_container_width=True)
    
    # Generate Post
    if generate_btn:
        if not topic or not topic.strip():
            st.warning("⚠️ Please enter a topic first!")
        elif use_custom and not selected_prompt.strip():
            st.warning("⚠️ Please provide custom instructions or uncheck the custom prompt option!")
        else:
            with st.spinner("✍️ Writing your post... This may take 10-20 seconds"):
                try:
                    post = agent.generate_post(
                        selected_prompt, 
                        topic, 
                        audience_desc, 
                        tone, 
                        length
                    )
                    st.session_state['current_post'] = post
                    st.success("✅ Post generated successfully!")
                except Exception as e:
                    st.error(f"❌ Error generating post: {str(e)}")
                    st.info("💡 Make sure your API key is valid and you have internet connection.")

    # Display and Edit Post
    if st.session_state.get('current_post'):
        st.markdown("---")
        
        # Metrics Dashboard
        st.markdown('<div class="section-badge">Post Analytics</div>', unsafe_allow_html=True)
        
        char_count = len(st.session_state.current_post)
        word_count = len(st.session_state.current_post.split())
        hashtag_count = st.session_state.current_post.count('#')
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            delta_text = f"{char_count - 3000}" if char_count > 3000 else f"{3000 - char_count} left"
            st.metric("Characters", f"{char_count:,}", delta_text)
        with metric_col2:
            st.metric("Words", word_count)
        with metric_col3:
            st.metric("Hashtags", hashtag_count)
        with metric_col4:
            progress = min(char_count / 3000, 1.0)
            st.metric("LinkedIn Limit", f"{progress*100:.1f}%")
        
        # Progress bar
        st.progress(progress)
        
        if char_count > 3000:
            st.warning(f"⚠️ Your post exceeds LinkedIn's 3000 character limit by {char_count - 3000} characters. Consider shortening it.")
        
        # Check word count against selected length
        word_count_limits = {"Short": 100, "Medium": 250, "Long": 500}
        if 'length' in locals() and length in word_count_limits:
            expected_limit = word_count_limits[length]
            if word_count > expected_limit * 1.2:
                st.info(f"ℹ️ This post has {word_count} words, which is above the {length.lower()} length target of ~{expected_limit} words. Consider trimming for better engagement.")
        
        st.markdown("---")
        
        # Edit Section
        st.markdown('<div class="section-badge">Edit & Refine</div>', unsafe_allow_html=True)
        edited_post = st.text_area(
            "Make any changes below:",
            value=st.session_state['current_post'],
            height=300,
            help="Edit the generated post to match your style",
            label_visibility="collapsed",
            key="edit_area"
        )
        
        # Action Buttons
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
        
        with btn_col1:
            if st.button("💾 Save Changes", use_container_width=True):
                st.session_state['current_post'] = edited_post
                st.success("✅ Post updated!")
        
        with btn_col2:
            if st.button("📋 Copy", use_container_width=True):
                if HAS_PYPERCLIP:
                    try:
                        pyperclip.copy(edited_post)
                        st.success("✅ Copied to clipboard!")
                    except Exception as e:
                        st.warning("⚠️ Clipboard copy failed. Use download button instead.")
                        st.code(edited_post, language=None)
                else:
                    st.code(edited_post, language=None)
                    st.info("💡 pyperclip not installed. Select and copy the text above, or use download.")
        
        with btn_col3:
            st.download_button(
                "📥 Download",
                edited_post,
                file_name=f"linkedin_post_{topic[:20].replace(' ', '_') if topic else 'post'}.txt",
                use_container_width=True
            )
        
        with btn_col4:
            if st.button("📌 Save to History", use_container_width=True):
                if edited_post not in st.session_state.post_history:
                    st.session_state.post_history.append(edited_post)
                    st.success(f"✅ Saved! ({len(st.session_state.post_history)} posts in history)")
                else:
                    st.info("ℹ️ This post is already in history")
        
        st.markdown("---")
        
        # LinkedIn Preview
        st.markdown('<div class="section-badge">LinkedIn Preview</div>', unsafe_allow_html=True)
        st.caption("How your post will look on LinkedIn")
        
        # Enhanced Preview Card
        preview_html = f"""
        <div class="preview-container">
            <div style="display: flex; align-items: center; margin-bottom: 16px; padding-bottom: 16px; border-bottom: 1px solid #e0e0e0;">
                <div style="
                    width: 56px; 
                    height: 56px; 
                    background: linear-gradient(135deg, #0a66c2, #004182); 
                    border-radius: 50%; 
                    margin-right: 12px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    color: white;
                    font-weight: bold;
                    font-size: 24px;
                    box-shadow: 0 2px 8px rgba(10, 102, 194, 0.3);
                ">
                    {topic[0].upper() if topic else 'Y'}
                </div>
                <div>
                    <div style="font-weight: 700; color: #000000; font-size: 16px; margin-bottom: 2px;">Your Profile Name</div>
                    <div style="font-size: 13px; color: #666666;">{audience_name} • Just now</div>
                </div>
            </div>
            <div style="
                white-space: pre-wrap; 
                color: #1d1d1f; 
                font-size: 15px; 
                line-height: 1.6; 
                margin-bottom: 16px;
            ">
                {edited_post}
            </div>
            <div style="
                display: flex; 
                justify-content: space-around; 
                padding-top: 12px; 
                border-top: 1px solid #e0e0e0;
                color: #666666;
                font-size: 14px;
                font-weight: 600;
            ">
                <span>👍 Like</span>
                <span>💬 Comment</span>
                <span>🔁 Repost</span>
                <span>📤 Send</span>
            </div>
        </div>
        """
        st.markdown(preview_html, unsafe_allow_html=True)
        
        # Post History Section
        if st.session_state.post_history:
            st.markdown("---")
            st.markdown('<div class="section-badge">Post History</div>', unsafe_allow_html=True)
            
            # Show last 5 posts
            recent_posts = list(reversed(st.session_state.post_history[-5:]))
            
            for idx, saved_post in enumerate(recent_posts):
                post_number = len(st.session_state.post_history) - idx
                col1, col2 = st.columns([5, 1])
                
                with col1:
                    preview_text = saved_post[:100] + "..." if len(saved_post) > 100 else saved_post
                    st.text(f"Post #{post_number}: {preview_text}")
                
                with col2:
                    if st.button(f"Load", key=f"load_{idx}", use_container_width=True):
                        st.session_state.current_post = saved_post
                        st.rerun()
            
            if len(st.session_state.post_history) > 5:
                st.caption(f"Showing last 5 of {len(st.session_state.post_history)} posts")

if __name__ == "__main__":
    main()