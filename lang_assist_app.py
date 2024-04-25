import streamlit as st

from lang_assist import LangAssist

st.set_page_config(
    page_title="Lang Assist",
    page_icon="📑",
    menu_items={
        # 'Get help': 'https://www.youtube.com',
        'About': 'https://github.com/Zhuohua-HUANG/LangAssist?tab=readme-ov-file',
        'Report a bug': "https://github.com/Zhuohua-HUANG/LangAssist/issues",
    }
)
try:
    # st.markdown("<h1 style='text-align: center; color: grey;'>Get :green[Job Aligned] :orange[Killer] Resume :sunglasses:</h1>", unsafe_allow_html=True)
    st.header("HKUST :green[Postgraduate Course] :orange[Inquiry Assistant]", divider='rainbow')

    # chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.chat_message("assistant"):
        st.write("Hello 👋 How can I help you? 🤗")

    # display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    lang_assist = LangAssist()

    # question example:
    # I am a prospective student and I want to learn a programme related to computer science. Can you compare the courses and differences between information technology and big data technology?
    question = st.chat_input("Ask any question about postgraduate courses...")

    if question:
        with st.chat_message("user"):
            st.write(question)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": question})

        # RAG response
        with st.chat_message("assistant"):
            response = st.write_stream(lang_assist.get_answer(question))
        st.session_state.messages.append({"role": "assistant", "content": response})

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.markdown("<h3 style='text-align: center;'>Please try again!</h3>", unsafe_allow_html=True)
    st.stop()
