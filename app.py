import streamlit as st
import time


from src.utils import * 

# Question: Can you wrap this around a front-end, or a terminal console interface ? (I think terminal console is easier here)
get_action = lambda: input("Do you like this paper? (y/n/back): ").lower()
get_reject_feedback = lambda: input("Please provide feedback for rejection: ")

def build_research_graph():
    """ 
    1. Scrape ArXiv for newest 200 AI papers 
    2. AI do ranking & analysis base on abstracts
    3. Human select 5 papers from top-ranked papers
    4. Conditional on human acceptance, proceed, otherwise, summarize human preference and go back to 2.
    -- Output 5 papers & analysis
    """
    model = ChatOpenAI(model="gpt-4o")
    builder = StateGraph(State)
    
    # Research graph part
    research_entry_point = "crawl_arxiv_node"
    builder.add_node(research_entry_point, CrawlNode(research_entry_point))
    builder.add_edge(START, research_entry_point)
    
    builder = add_selection_nodes(builder, research_entry_point, "human_approval_node", model)
    
    # Human approval graph part
    human_approval_node = "human_approval_node"
    builder.add_node(human_approval_node, HumanApprovalNode(human_approval_node,
                                                            get_action=get_action,
                                                            get_reject_feedback=get_reject_feedback))  # Assuming HumanApprovalNode is defined elsewhere
    
    # So here we ought to have a conditional edge, if there is "redo" in the message, we re-run from research_entry_point
    builder.add_conditional_edges(human_approval_node, 
                                  should_continue,
                                  {
                                    "continue": END,
                                    "redo": research_entry_point,
                                  },)
        
    return builder.compile()

def mock_fetch_topics():
    # Simulating API call to fetch trending topics
    time.sleep(1.5)  # Simulate network delay
    return [
        'The Impact of AI on Job Markets',
        'Climate Change: Recent Developments',
        'Advances in Quantum Computing',
        'The Future of Remote Work',
        'Blockchain Beyond Cryptocurrency'
    ]

def main():
    st.title("Interactive Article Writer")

    # Initialize session state variables
    if 'step' not in st.session_state:
        st.session_state.step = 0
    if 'topics' not in st.session_state:
        st.session_state.topics = []
    if 'selected_topic' not in st.session_state:
        st.session_state.selected_topic = None

    # Sidebar for progress tracking
    with st.sidebar:
        st.header("Progress")
        steps = ['Research', 'Topic Selection', 'Writing', 'Review', 'Publish']
        for i, step in enumerate(steps):
            if i == st.session_state.step:
                st.markdown(f"**{step}**")
            else:
                st.write(step)

    # Main content area
    if st.session_state.step == 0:
        st.header("Research")
        st.write("AI is navigating the web to find trending topics...")
        if st.button("Start Research"):
            with st.spinner("Researching..."):
                st.session_state.topics = mock_fetch_topics()
            st.session_state.step = 1
            st.experimental_rerun()

    elif st.session_state.step == 1:
        st.header("Topic Selection")
        for topic in st.session_state.topics:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(topic)
            with col2:
                if st.button("Select", key=topic):
                    st.session_state.selected_topic = topic
                    st.session_state.step = 2
                    st.experimental_rerun()

    elif st.session_state.step == 2:
        st.header("Writing")
        st.write(f"AI is writing an article on: {st.session_state.selected_topic}")
        if st.button("Review Draft"):
            st.session_state.step = 3
            st.experimental_rerun()

    elif st.session_state.step == 3:
        st.header("Review")
        st.write("Please review the article and provide feedback:")
        feedback = st.text_area("Your feedback")
        if st.button("Submit Feedback"):
            if feedback:
                st.success("Feedback submitted successfully!")
                st.session_state.step = 4
                st.experimental_rerun()
            else:
                st.warning("Please provide feedback before submitting.")

    elif st.session_state.step == 4:
        st.header("Publish")
        st.write("Your article is ready to be published!")
        if st.button("Publish Article"):
            st.success("Article published successfully!")

if __name__ == "__main__":
    main()