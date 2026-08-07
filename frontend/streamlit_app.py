import requests
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime


BACKEND_URL = "http://127.0.0.1:8000"


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="FinDoc AI",
    page_icon="📊",
    layout="wide"
)


# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown(
"""
<style>

.main {
    background-color:#f8fafc;
}


.block-container {
    padding-top:2rem;
}


.card {

    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0 4px 12px rgba(0,0,0,0.08);

}


.metric-card {

    background:linear-gradient(
        135deg,
        #0f172a,
        #1e40af
    );

    color:white;
    padding:20px;
    border-radius:15px;

}


.title {

    font-size:38px;
    font-weight:700;

}


.subtitle {

    color:#64748b;
    font-size:18px;

}


.chat-user {

    background:#dbeafe;
    padding:12px;
    border-radius:12px;

}


.chat-ai {

    background:#ecfdf5;
    padding:12px;
    border-radius:12px;

}



</style>
""",
unsafe_allow_html=True
)



# =====================================================
# HEADER
# =====================================================


st.markdown(
"""
<div class="title">
📊 FinDoc AI
</div>

<div class="subtitle">
AI-powered Financial Document Intelligence Platform
</div>

<br>

""",
unsafe_allow_html=True
)



# =====================================================
# SIDEBAR
# =====================================================


with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/3135/3135706.png",
        width=80
    )


    st.title("FinDoc AI")


    st.caption(
        "Incremental RAG System for Financial Reports"
    )


    st.divider()


    page = st.radio(

        "Navigation",

        [

            "📄 Upload Report",

            "💬 Financial Assistant",

            "📈 Analytics Dashboard"

        ]

    )


    st.divider()


    st.success(
        "Backend Connected"
    )



# =====================================================
# UPLOAD
# =====================================================


if page == "📄 Upload Report":


    st.header(
        "Upload Financial Report"
    )


    col1,col2 = st.columns(
        [2,1]
    )


    with col1:


        file = st.file_uploader(

            "Upload annual report / financial statement",

            type=["pdf"]

        )


        if file:


            st.info(
                f"""
                File: {file.name}

                Size:
                {round(file.size/1024,2)} KB
                """
            )


            if st.button(
                "🚀 Process & Index",
                use_container_width=True
            ):


                with st.spinner(
                    "Extracting, chunking and indexing..."
                ):


                    response=requests.post(

                        f"{BACKEND_URL}/upload",

                        files={

                            "file":
                            (
                                file.name,
                                file,
                                "application/pdf"
                            )

                        }

                    )


                if response.status_code==200:


                    st.success(
                        "Document indexed successfully"
                    )


                    result=response.json()


                    st.json(result)



                else:

                    st.error(
                        response.text
                    )



    with col2:


        st.markdown(

        """
        <div class="card">

        ### Supported Documents

        ✔ Annual Reports

        ✔ Balance Sheets

        ✔ Income Statements

        ✔ Investor Presentations

        ✔ Financial Filings


        </div>

        """,

        unsafe_allow_html=True

        )



# =====================================================
# CHATBOT
# =====================================================


elif page=="💬 Financial Assistant":



    st.header(
        "💬 Financial AI Assistant"
    )


    if "messages" not in st.session_state:

        st.session_state.messages=[]



    for msg in st.session_state.messages:


        with st.chat_message(
            msg["role"]
        ):

            st.write(
                msg["content"]
            )



    question=st.chat_input(

        "Ask about revenue, profit, risks..."

    )



    if question:


        st.session_state.messages.append(

            {

                "role":"user",

                "content":question

            }

        )



        with st.chat_message("user"):

            st.write(question)



        with st.chat_message("assistant"):


            with st.spinner(
                "Analyzing financial documents..."
            ):


                response=requests.post(

                    f"{BACKEND_URL}/chat/",

                    json={

                        "question":question

                    }

                )


            if response.status_code==200:


                data=response.json()


                answer=data["answer"]


                st.write(answer)



                st.session_state.messages.append(

                    {

                        "role":"assistant",

                        "content":answer

                    }

                )


                with st.expander(
                    "📚 Retrieved Evidence"
                ):


                    for s in data["sources"]:


                        st.markdown(

                        f"""

                        **{s['document']}**

                        Page:
                        {s['page']}

                        Similarity:
                        {s['distance']:.4f}


                        """

                        )


                        st.write(
                            s["text"][:400]
                        )


                        st.divider()



            else:

                st.error(
                    response.text
                )





# =====================================================
# ANALYTICS DASHBOARD
# =====================================================



elif page=="📈 Analytics Dashboard":


    st.header(
        "Financial Intelligence Dashboard"
    )


    response=requests.get(

        f"{BACKEND_URL}/analytics"

    )


    if response.status_code!=200:

        st.error(
            "Analytics unavailable"
        )

    else:


        data=response.json()


        documents=data["documents"]

        incremental=data["incremental"]

        queries=data["queries"]



        # -----------------------------
        # KPI CARDS
        # -----------------------------


        c1,c2,c3,c4=st.columns(4)



        c1.metric(

            "Reports Indexed",

            documents["reports"]

        )


        c2.metric(

            "Companies",

            documents["companies"]

        )


        c3.metric(

            "Total Chunks",

            documents["chunks"]

        )


        c4.metric(

            "Queries",

            queries["total_queries"]

        )



        st.divider()



        tab1,tab2,tab3=st.tabs(

            [

                "Indexing",

                "Performance",

                "Query Analytics"

            ]

        )



        # -----------------------------
        # INDEXING
        # -----------------------------


        with tab1:


            df=pd.DataFrame(

            {

            "Metric":

            [

            "New",

            "Modified",

            "Deleted",

            "Reused"

            ],


            "Count":

            [

            incremental["new_chunks"],

            incremental["modified_chunks"],

            incremental["deleted_chunks"],

            incremental["unchanged_chunks"]

            ]

            }


            )


            fig=px.bar(

                df,

                x="Metric",

                y="Count",

                title="Incremental Chunk Updates"

            )


            st.plotly_chart(

                fig,

                use_container_width=True

            )



        # -----------------------------
        # PERFORMANCE
        # -----------------------------


        with tab2:


            df=pd.DataFrame(

            {

            "Metric":

            [

            "Embedding Calls",

            "Saved Calls"

            ],


            "Value":

            [

            incremental["embedding_calls"],

            incremental["embedding_calls_saved"]

            ]

            }

            )



            fig=px.pie(

                df,

                names="Metric",

                values="Value",

                title="Embedding Optimization"

            )


            st.plotly_chart(

                fig,

                use_container_width=True

            )


            st.metric(

                "Avg Processing Time",

                f"{incremental['average_processing_time']} sec"

            )



        # -----------------------------
        # QUERY
        # -----------------------------


        with tab3:


            st.metric(

                "Average Response Time",

                f"{queries['average_response_time']} sec"

            )


            st.progress(

                min(
                    queries["average_response_time"]/10,
                    1
                )

            )


