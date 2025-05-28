# CUSTOMER SERVICE CHATBOT API 
This project is a chatbot for an intelligent customer service chatbot. It uses a fine-tuned SentenceTransformer- based intent classification model, semantic search for a knowledge base, and logic for handling sales leads, technical support, and product feature requests. 

 🚀 FEATURES 

  - Fine-tuned intent classifier using SentenceTransformers + MLP head 
  - Semantic search over knowledge base article using FAISS
  - Handles: 
    - Technical Support queries 
    - Product Feature Requests 
    - Sales Lead information extraction 
  - Sentiment analysis via TextBlob
  - Automatic escalation logging and context tracking

<details>
<summary>📁 Project Structure</summary>

project-root/
├── backend/
│   ├── main.py
│   ├── model_training/
│   │   └── fine_tuned_intent_bert/
│   ├── customer_intent_dataset.jsonl
│   ├── kb.json
│   ├── feature_requests.json
│   ├── sales_leads.json
│   ├── negative_feedback.json
│   └── unresolved_technical_queries.json
├── frontend/
│   ├── public/
│   ├── src/
│   ├── package.json
│   └── ...
└── README.md
</details>

⚙️ PREREQUISITES

   Please refer to the requirements.txt(backend) and requirement.txt(frontend) file in the 
   folder having all the prerequisites for the project. 

## 🖥 LOCAL DEPLOYMENT INSTRUCTIONS 

    1. Ensure the following are installed: 
       - Python 3.8+
       - A python environment manager 

    2. 📦 INSTALL DEPENDENCIES 
        -pip install -r requirements.txt 
        - Initialize TextBlob data (once) 
          python -m textblob.download_corpora 

    

     3. 🧠 MODEL PREPARATION 
     The file intent_classifier_sentence_transformers.pt is present in the root
     directory as stated in the root directory above. This is the trained model for intent 
     classification. The model_training.py (housing the training of the model) is also in 
     model_training subfolder in the root directory.  

     
     4. 📚 Knowledge Base File
      kb.json file houses all the knowledge base with some general prompts relevant to the    
      companies 

     
     5. 🏃‍♂️ RUN THE SERVER 
        bash: 
            uvicorn main:app --reload 

        This will start the FastAPI server at: 
        http://127.0.0.1:8000

     6. 🌐 FRONTEND SETUP
         - In a different command prompt window
         - Navigating to the frontend directory 
         bash: 
             cd../frontend 
         
         - Install dependencies 
           npm install 
           npx install 
           npm install axios

         - Run the frontend dev server 
           npm start 
         - This frontend will be available at: http://localhost:3000
           
         
      7.📡 TESTING 
        - Open the browser and go to http://localhost:3000
        - While interacting with the chatbot it can be observed that 


       8. TIPS: 
        - The chatbot will log feature requests, sales leads, and unresolved tech queries 
          into JSON files. 
        - If a user expresses strong negative sentiment, the system will escalate and log 
          log their message for review in separate JSON file. 

