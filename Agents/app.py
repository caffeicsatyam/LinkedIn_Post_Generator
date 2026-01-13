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
    initial_sidebar_state="expanded"
)

# Enhanced CSS for modern, professional look
st.markdown("""
    <style>
    /* Main background */
    .main {
        background-color: #f3f2ef;
        padding: 2rem 1rem;
    }
    
    /* Header styling */
    h1 {
        color: #0a66c2;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
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
    }
    
    .stTextInput>div>div>input:focus,
    .stTextArea>div>div>textarea:focus {
        border-color: #0a66c2;
        box-shadow: 0 0 0 3px rgba(10, 102, 194, 0.1);
    }
    
    /* Selectbox */
    .stSelectbox>div>div>select {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
    }
    
    /* Sidebar - Dark mode compatible */
    [data-testid="stSidebar"] {
        background-color: #262730 !important;
        border-right: 1px solid #3a3a4a;
    }
    
    /* Sidebar text colors for dark mode */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: #fafafa !important;
    }
    
    /* Sidebar selectbox and inputs */
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stTextArea label,
    [data-testid="stSidebar"] .stCheckbox label {
        color: #fafafa !important;
    }
    
    /* Sidebar selectbox dropdown */
    [data-testid="stSidebar"] .stSelectbox>div>div>select {
        background-color: #1e1e2e !important;
        color: #fafafa !important;
        border-color: #3a3a4a !important;
    }
    
    /* Sidebar text inputs */
    [data-testid="stSidebar"] .stTextInput>div>div>input,
    [data-testid="stSidebar"] .stTextArea>div>div>textarea {
        background-color: #1e1e2e !important;
        color: #fafafa !important;
        border-color: #3a3a4a !important;
    }
    
    /* Sidebar radio buttons */
    [data-testid="stSidebar"] .stRadio label {
        color: #fafafa !important;
    }
    
    /* Sidebar checkbox */
    [data-testid="stSidebar"] .stCheckbox label {
        color: #fafafa !important;
    }
    
    /* Sidebar caption */
    [data-testid="stSidebar"] .stCaption {
        color: #b0b0b0 !important;
    }
    
    /* Sidebar expander */
    [data-testid="stSidebar"] .streamlit-expanderHeader {
        color: #fafafa !important;
    }
    
    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton>button {
        background-color: #0a66c2 !important;
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stButton>button:hover {
        background-color: #004182 !important;
    }
    
    /* Sidebar divider */
    [data-testid="stSidebar"] hr {
        border-color: #3a3a4a !important;
    }
    
    /* Cards and containers */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Success/Info messages */
    .stSuccess {
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stWarning {
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Spinner */
    .stSpinner {
        margin-top: 2rem;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 2px solid #e0e0e0;
    }
    
    /* Custom badge */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: #e8f0fe;
        color: #0a66c2;
    }
    
    /* Copy button styling */
    .copy-btn {
        background-color: #f3f2ef;
        color: #0a66c2;
        border: 2px solid #0a66c2;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        cursor: pointer;
    }
    
    .copy-btn:hover {
        background-color: #0a66c2;
        color: white;
    }
    
    /* Section headers with visual indicators */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .section-icon {
        width: 24px;
        height: 24px;
        background: linear-gradient(135deg, #0a66c2, #004182);
        border-radius: 6px;
        display: inline-block;
    }
    
    /* Status indicators */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-success {
        background-color: #d4edda;
        color: #155724;
    }
    
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
    }
    
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
    }
    
    /* Action button icons using CSS */
    .btn-icon {
        display: inline-block;
        width: 16px;
        height: 16px;
        margin-right: 6px;
        vertical-align: middle;
    }
    
    /* Visual separator */
    .visual-separator {
        height: 2px;
        background: linear-gradient(90deg, transparent, #0a66c2, transparent);
        margin: 1.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

class LinkedInPostAgent:
    """Interactive agent for generating LinkedIn posts with Streamlit UI"""
    
    def __init__(self, api_key: Optional[str] = None):
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
        
        # Keep the model as specified in original code
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0.9,
            max_output_tokens=2048
        )
        
        self.prompt_templates = self._load_prompt_templates()
        self.audiences = self._load_audiences()

    def _load_prompt_templates(self) -> Dict[str, Dict]:
        return {
            "Personal Story": "Write a LinkedIn post sharing a personal story about {topic}. Start with a relatable moment or challenge. Build narrative with details. Share growth. End with takeaway. Authentic and vulnerable.",
            "Quick Tips List": "Write a LinkedIn post with actionable tips about {topic}. Bold hook. List 3-5 specific tips with examples. Use emojis. End with a question.",
            "Controversial Opinion": "Challenge common thinking about {topic}. Open with a hot take. Explain why conventional wisdom is wrong. Support with logic. End with a thought-provoking question.",
            "Behind-the-Scenes": "Reveal behind-the-scenes insights about {topic}. Share insider knowledge. Include processes. Explain why it matters. Transparent and authentic.",
            "Trend Analysis": "Analyze a trend related to {topic}. Driving forces, impact, and predictions. How should the audience respond?",
            "Motivational Message": "Uplifting post about {topic}. Powerful quote/statement. Share a setback and transformation. Empowering call to action."
        }
    
    def _load_audiences(self) -> Dict[str, str]:
        return {
            "Startup Founders": "Entrepreneurs building tech startups, interested in growth and scaling",
            "Marketing Professionals": "Marketers focused on strategy and brand growth",
            "Software Developers": "Engineers and developers interested in tech and career growth",
            "Sales Professionals": "Sales people focused on prospecting and deals",
            "Job Seekers": "Professionals seeking career opportunities",
            "Business Leaders": "Executives and managers interested in leadership",
            "Freelancers": "Independent professionals building their brand",
            "General Professional": "Broad professional audience across industries"
        }

    def _humanization_rules(self) -> str:
        return """
            HUMANIZATION RULES:
            - Conversational tone, use "I", "you", "we".
            - Short paragraphs (2-3 lines).
            - Simple language, no buzzwords (leverage, synergy).
            - Show personality and vary sentence length.
        """

    def generate_post(self, prompt: str, topic: str, audience: str, tone: str = "Professional", length: str = "Medium") -> str:
        tone_instructions = {
            "Professional": "Use a formal, business-appropriate tone. Professional language but still engaging.",
            "Casual": "Use a relaxed, conversational tone. Friendly and approachable like talking to a colleague.",
            "Controversial": "Use a bold, thought-provoking tone. Challenge conventional wisdom respectfully.",
            "Empathetic": "Use a warm, understanding tone. Show emotional intelligence and connection."
        }
        
        length_instructions = {
            "Short": "Keep it concise - 2-3 short paragraphs, maximum 150 words.",
            "Medium": "Standard length - 3-5 paragraphs, around 200-300 words.",
            "Long": "Detailed storytelling - 5+ paragraphs, 400+ words for deep engagement."
        }
        
        final_prompt = PromptTemplate(
            input_variables=["prompt", "topic", "audience", "rules", "tone", "length"],
            template="""
                You are an expert LinkedIn content creator known for viral, high-engagement posts.
                
                USER'S INSTRUCTIONS: {prompt}
                TOPIC: {topic}
                TARGET AUDIENCE: {audience}
                TONE: {tone}
                LENGTH: {length}
                
                {rules}
                
                FORMATTING REQUIREMENTS:
                - Start with a strong, attention-grabbing hook (first 1-2 lines)
                - Use strategic line breaks every 2-3 sentences for readability
                - Include 3-5 relevant hashtags at the end
                - End with an engaging call-to-action or question
                
                Make it sound authentically human, not AI-generated.
            """
        )
        chain = final_prompt | self.llm
        result = chain.invoke({
            "prompt": prompt,
            "topic": topic,
            "audience": audience,
            "rules": self._humanization_rules(),
            "tone": tone_instructions.get(tone, tone_instructions["Professional"]),
            "length": length_instructions.get(length, length_instructions["Medium"])
        })
        return result.content if hasattr(result, 'content') else str(result)

# --- MAIN APP FLOW ---

def main():
    # Initialize session state
    if 'post_history' not in st.session_state:
        st.session_state.post_history = []
    if 'current_post' not in st.session_state:
        st.session_state.current_post = ""
    
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        st.error("API Key not found! Please set GOOGLE_API_KEY in your .env file.")
        st.stop()
    
    try:
        agent = LinkedInPostAgent(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing agent: {str(e)}")
        st.stop()

    # Sidebar Configuration
    with st.sidebar:
        st.markdown('<div class="section-header"><div class="section-icon"></div><h3>Post Configuration</h3></div>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Audience Selection
        st.markdown("**Target Audience**")
        audience_name = st.selectbox(
            "Who is your audience?", 
            list(agent.audiences.keys()),
            help="Select the primary audience for your post"
        )
        audience_desc = agent.audiences[audience_name]
        st.caption(f"*{audience_desc}*")
        
        st.markdown("---")
        
        # Tone Selection
        st.markdown("**Content Tone**")
        tone = st.radio(
            "Select tone:",
            ["Professional", "Casual", "Controversial", "Empathetic"],
            horizontal=False
        )
        
        st.markdown("---")
        
        # Length Selection
        st.markdown("**Post Length**")
        length = st.radio(
            "Select length:",
            ["Short", "Medium", "Long"],
            index=1,
            horizontal=False
        )
        
        st.markdown("---")
        
        # Template Selection
        st.markdown("**Content Style**")
        use_custom = st.checkbox("Use custom prompt", help="Check to write your own instructions")
        
        if not use_custom:
            template_name = st.selectbox(
                "Choose a template:", 
                list(agent.prompt_templates.keys()),
                help="Select a pre-built template style"
            )
            selected_prompt = agent.prompt_templates[template_name]
            with st.expander("Template Preview"):
                st.caption(selected_prompt)
        else:
            selected_prompt = st.text_area(
                "Custom instructions:", 
                placeholder="e.g., Write a funny, engaging post about coding that includes a personal anecdote...",
                height=100
            )
        
        st.markdown("---")
        
        # Post History
        if st.session_state.post_history:
            st.markdown('<div class="section-header"><div class="section-icon"></div><h3>Post History</h3></div>', unsafe_allow_html=True)
            for idx, saved_post in enumerate(reversed(st.session_state.post_history[-5:]), 1):
                if st.button(f"Load Post #{len(st.session_state.post_history) - len(st.session_state.post_history[-5:]) + idx}", 
                           key=f"load_{idx}", use_container_width=True):
                    st.session_state.current_post = saved_post
                    st.rerun()

    # Main Content Area
    st.title("LinkedIn Post Generator")
    st.markdown("**Generate high-engagement, human-sounding posts in seconds.**")
    st.markdown('<div class="visual-separator"></div>', unsafe_allow_html=True)
    
    # Topic Input
    col1, col2 = st.columns([3, 1])
    with col1:
        topic = st.text_input(
            "**What is your post about?**", 
            placeholder="e.g., The future of Remote Work in 2026",
            help="Enter the main topic or theme for your LinkedIn post"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        generate_btn = st.button("Generate Post", type="primary", use_container_width=True)
    
    # Generate Post
    if generate_btn:
        if not topic or not topic.strip():
            st.warning("Please enter a topic first!")
        elif use_custom and not selected_prompt.strip():
            st.warning("Please provide custom instructions or uncheck the custom prompt option!")
        else:
            with st.spinner("Writing your post... This may take 10-20 seconds"):
                try:
                    post = agent.generate_post(selected_prompt, topic, audience_desc, tone, length)
                    st.session_state['current_post'] = post
                    st.success("Post generated successfully!")
                    st.markdown('<span class="status-badge status-success">SUCCESS</span>', unsafe_allow_html=True)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating post: {str(e)}")
                    st.info("Make sure your API key is valid and you have internet connection.")

    # Display and Edit Post
    if st.session_state.get('current_post'):
        st.markdown("---")
        
        # Metrics
        char_count = len(st.session_state.current_post)
        word_count = len(st.session_state.current_post.split())
        hashtag_count = st.session_state.current_post.count('#')
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            st.metric("Characters", f"{char_count:,}", f"{char_count - 3000}" if char_count > 3000 else f"{3000 - char_count} remaining")
        with metric_col2:
            st.metric("Words", word_count)
        with metric_col3:
            st.metric("Hashtags", hashtag_count)
        with metric_col4:
            progress = min(char_count / 3000, 1.0)
            st.metric("LinkedIn Limit", f"{progress*100:.1f}%")
            st.progress(progress)
        
        st.markdown("---")
        
        # Edit Section
        st.markdown('<div class="section-header"><div class="section-icon"></div><h2>Edit Your Post</h2></div>', unsafe_allow_html=True)
        edited_post = st.text_area(
            "Make any changes below:",
            value=st.session_state['current_post'],
            height=300,
            help="Edit the generated post to match your style"
        )
        
        # Action Buttons
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
        
        with btn_col1:
            if st.button("Save Changes", use_container_width=True):
                st.session_state['current_post'] = edited_post
                st.success("Post updated!")
                st.rerun()
        
        with btn_col2:
            if st.button("Copy to Clipboard", use_container_width=True):
                if HAS_PYPERCLIP:
                    try:
                        pyperclip.copy(edited_post)
                        st.success("Copied to clipboard!")
                    except:
                        st.warning("Clipboard copy failed. Use download button instead.")
                else:
                    # Fallback: show text in code block for easy copying
                    st.code(edited_post, language=None)
                    st.info("Select and copy the text above, or use the download button.")
        
        with btn_col3:
            st.download_button(
                "Download .txt",
                edited_post,
                file_name=f"linkedin_post_{topic[:20].replace(' ', '_')}.txt",
                use_container_width=True
            )
        
        with btn_col4:
            if st.button("Save to History", use_container_width=True):
                st.session_state.post_history.append(edited_post)
                st.success(f"Saved! ({len(st.session_state.post_history)} posts in history)")
        
        st.markdown("---")
        
        # LinkedIn Preview
        st.markdown('<div class="section-header"><div class="section-icon"></div><h2>LinkedIn Preview</h2></div>', unsafe_allow_html=True)
        st.caption("How your post will look on LinkedIn")
        
        # Enhanced Preview Card
        preview_html = f"""
        <div style="
            background-color: #ffffff; 
            padding: 24px; 
            border-radius: 12px; 
            border: 1px solid #e0e0e0; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
            max-width: 600px;
            margin: 0 auto;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        ">
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
                <span>Like</span>
                <span>Comment</span>
                <span>Repost</span>
                <span>Send</span>
            </div>
        </div>
        """
        st.markdown(preview_html, unsafe_allow_html=True)
        
        # Warning if over limit
        if char_count > 3000:
            st.warning(f"Your post exceeds LinkedIn's 3000 character limit by {char_count - 3000} characters. Consider shortening it.")

if __name__ == "__main__":
    main()