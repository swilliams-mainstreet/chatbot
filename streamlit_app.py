import streamlit as st
import openai
import pinecone
import os


# Check if environment variables are present. If not, throw an error
if os.getenv('PINECONE_API_KEY') is None:
    st.error("PINECONE_API_KEY not set. Please set this environment variable and restart the app.")
if os.getenv('PINECONE_ENVIRONMENT') is None:
    st.error("PINECONE_ENVIRONMENT not set. Please set this environment variable and restart the app.")
if os.getenv('PINECONE_INDEX') is None:
    st.error("PINECONE_INDEX not set. Please set this environment variable and restart the app.")
if os.getenv('AZURE_OPENAI_APIKEY') is None:
    st.error("AZURE_OPENAI_APIKEY not set. Please set this environment variable and restart the app.")
if os.getenv('AZURE_OPENAI_BASE_URI') is None:
    st.error("AZURE_OPENAI_BASE_URI not set. Please set this environment variable and restart the app.")
if os.getenv('AZURE_OPENAI_EMBEDDINGS_MODEL_NAME') is None:
    st.error("AZURE_OPENAI_EMBEDDINGS_MODEL_NAME not set. Please set this environment variable and restart the app.")
if os.getenv('AZURE_OPENAI_GPT4_MODEL_NAME') is None:
    st.error("AZURE_OPENAI_GPT4_MODEL_NAME not set. Please set this environment variable and restart the app.")


# Add the UI title, text input and search button
st.title("MainStreet Family Care Q&A App 🔎")

role = st.selectbox(
    'What is your role?',
    ('Front Desk (FD)', 'Medical Assistant 1 (MA1)', 'Medical Assistant 2 (MA2)', 'Provider'))

if role == 'Front Desk (FD)': plain_enlgish_role = 'front desk employee (FD)'
elif role == 'Medical Assistant 1 (MA1)': plain_enlgish_role = 'medical assistant 1 (MA1)'
elif role == 'Medical Assistant 2 (MA2)': plain_enlgish_role = 'medical assistant 2 (MA2)'
elif role == 'Provider': plain_enlgish_role = 'provider'

query = st.text_input("What do you want to know?")

if st.button("Search"):
   
    # Get Pinecone API environment variables
    pinecone_api = os.getenv('PINECONE_API_KEY')
    pinecone_env = os.getenv('PINECONE_ENVIRONMENT')
    pinecone_index = os.getenv('PINECONE_INDEX')
    
    # Get Azure OpenAI environment variables
    openai.api_key = os.getenv('AZURE_OPENAI_APIKEY')
    openai.api_base = os.getenv('AZURE_OPENAI_BASE_URI')
    openai.api_type = 'azure'
    openai.api_version = '2023-03-15-preview'
    embeddings_model_name = os.getenv('AZURE_OPENAI_EMBEDDINGS_MODEL_NAME')
    gpt4_model_name = os.getenv('AZURE_OPENAI_GPT4_MODEL_NAME')
 
    # Initialize Pinecone client and set index
    pinecone.init(api_key=pinecone_api, environment=pinecone_env)
    index = pinecone.Index(pinecone_index)
 
    # Convert your query into a vector using Azure OpenAI
    try:
        query_vector = openai.Embedding.create(
            input=query,
            engine=embeddings_model_name,
        )["data"][0]["embedding"]
    except Exception as e:
        st.error(f"Error calling OpenAI Embedding API: {e}")
        st.stop()
 
    # Search for the 3 most similar vectors in Pinecone
    search_response = index.query(
        top_k=3,
        vector=query_vector,
        include_metadata=True)

    # Combine the 3 vectors into a single text variable that it will be added in the prompt
    chunks = [item["metadata"]['text'] for item in search_response['matches']]
    joined_chunks = "\n".join(chunks)

    # Write which are the selected chunks in the UI for debugging purposes
    with st.expander("Chunks"):
        for i, t in enumerate(chunks):
            t = t.replace("\n", " ")
            st.write("Chunk ", i, " - ", t)
    
    with st.spinner("Summarizing..."):
        try:
            # Build the prompt
            prompt = f"""
            Answer the following question based on the context below. The question is being asked from the perspective of an employee whose role is "{plain_enlgish_role}".
            If you don't know the answer, just say that you don't know. Don't try to make up an answer. Do not answer beyond this context.
            DO NOT start your response with "As a {plain_enlgish_role} employee," or anything similar, or I will kill you.
            DO NOT start your reponse with "Based on the context provided" or anything similar, or I will kill you.
            ---
            QUESTION: As a {plain_enlgish_role}, {query}                                            
            ---
            CONTEXT:
            {joined_chunks}
            """
 
            # Run chat completion using GPT-4
            response = openai.ChatCompletion.create(
                engine=gpt4_model_name,
                messages=[
                    { "role": "system", "content":  "You are a Q&A assistant." },
                    { "role": "user", "content": prompt }
                ],
                temperature=0.7,
                max_tokens=1000
            )
 
            # Get the response from GPT-4
            st.markdown("### Answer:")
            st.write(response.choices[0]['message']['content'])
   
   
        except Exception as e:
            st.error(f"Error with OpenAI Chat Completion: {e}")
