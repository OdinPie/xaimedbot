# enhanced_medical_chatbot_with_llm_questions.py

import os
import pandas as pd
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_openai import AzureChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import PyPDF2
import json
import pandas as pd
import numpy as np
import torch
from sentence_transformers import util

SYSTEM_READY = False

def visualize_disease_symptom_attribution(symptom_tracker, df_grouped, embedding_model):
    """
    Create a disease–symptom heatmap showing how much each user symptom
    contributes to each disease based on embedding similarity.
    """

    # 1. Collect user symptoms
    user_symptoms = symptom_tracker.get_all_positive_symptoms()
    if not user_symptoms:
        print("⚠️ No symptoms to visualize yet.")
        return

    # 2. Encode user symptoms
    symptom_embeds = embedding_model.encode(user_symptoms, convert_to_tensor=True)

    # 3. Get top diseases from current tracking
    if not symptom_tracker.disease_scores:
        print("⚠️ No disease scores found.")
        return

    top_diseases = [d for d, _ in sorted(symptom_tracker.disease_scores.items(), key=lambda x: x[1], reverse=True)[:5]]

    # 4. Compute similarity between each symptom and each disease embedding
    heatmap_data = []
    for disease in top_diseases:
        row = df_grouped[df_grouped['disease'] == disease]
        if row.empty:
            continue

        disease_sym_embeds = row.iloc[0]['symptom_embedding']

        # Represent disease as mean embedding of its known symptoms
        disease_vector = np.mean(disease_sym_embeds, axis=0, keepdims=True)

        # Cosine similarity: each symptom vs disease vector
        sims = util.cos_sim(symptom_embeds, torch.tensor(disease_vector)).cpu().numpy().flatten()
        heatmap_data.append(sims)

    if not heatmap_data:
        print("⚠️ No heatmap data could be created.")
        return

    # 5. Build DataFrame for visualization
    heatmap_df = pd.DataFrame(
        np.array(heatmap_data).T,
        index=user_symptoms,
        columns=top_diseases
    )

    



# --- Step 1: Enhanced PDF Data Extraction and Processing ---
def extract_pdf_content(pdf_path):
    """Extract text content from PDF"""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def parse_disease_data_from_pdf(pdf_content):
    """Parse complete disease information from PDF including symptoms, risk factors, and threshold values"""
    diseases_data = []
    disease_info = {}  # Store complete disease information
    
    # Split content by disease sections
    disease_sections = re.split(r'Disease Name:\s*([^\n]+)', pdf_content)[1:]
    
    for i in range(0, len(disease_sections), 2):
        if i + 1 < len(disease_sections):
            disease_name = disease_sections[i].strip()
            disease_content = disease_sections[i + 1]
            
            # Initialize disease entry
            disease_info[disease_name] = {
                'symptoms': [],
                'risk_factors': [],
                'threshold_values': [],
                'diagnostic_criteria': ''
            }
            
            # Extract symptoms
            symptoms_match = re.search(
                r'Symptoms:\s*-\s*(.*?)(?=Risk Factors:|Diagnostic Criteria:|$)',
                disease_content, re.DOTALL
            )
            if symptoms_match:
                symptoms_text = symptoms_match.group(1).strip()
                symptoms = parse_bullet_points(symptoms_text)
                disease_info[disease_name]['symptoms'] = symptoms
                
                # Add symptoms to main data structure
                for symptom in symptoms:
                    weight = calculate_symptom_weight(symptom)
                    diseases_data.append({
                        'disease': disease_name,
                        'symptom': symptom,
                        'weight': weight,
                        'evidence_source': 'PDF Medical Guidelines'
                    })
            
            # Extract risk factors
            risk_factors_match = re.search(
                r'Risk Factors:\s*-\s*(.*?)(?=Diagnostic Criteria:|Threshold Values:|$)',
                disease_content, re.DOTALL
            )
            if risk_factors_match:
                risk_factors_text = risk_factors_match.group(1).strip()
                risk_factors = parse_bullet_points(risk_factors_text)
                disease_info[disease_name]['risk_factors'] = risk_factors
            
            # Extract diagnostic criteria
            diagnostic_match = re.search(
                r'Diagnostic Criteria:\s*-\s*(.*?)(?=Threshold Values:|Basic Management:|$)',
                disease_content, re.DOTALL
            )
            if diagnostic_match:
                disease_info[disease_name]['diagnostic_criteria'] = diagnostic_match.group(1).strip()
            
            # Extract threshold values
            threshold_match = re.search(
                r'Threshold Values:\s*(?:-\s*)?(.*?)(?=Basic Management:|When to Seek|$)',
                disease_content, re.DOTALL
            )
            if threshold_match:
                threshold_text = threshold_match.group(1).strip()
                thresholds = parse_threshold_values(threshold_text)
                disease_info[disease_name]['threshold_values'] = thresholds
    
    return diseases_data, disease_info

def parse_bullet_points(text):
    """Parse bullet-pointed text into a list"""
    items = []
    
    # Split by bullet points
    bullet_items = re.split(r'[-•]\s*', text)
    
    for item in bullet_items:
        item = item.strip().rstrip('.')
        if not item or len(item) < 3:
            continue
        
        # Split by 'and' to separate compound items
        and_parts = re.split(r'\s+and\s+', item, flags=re.IGNORECASE)
        
        for part in and_parts:
            # Split by commas and semicolons
            comma_parts = re.split(r'[,;]\s*', part)
            
            for subitem in comma_parts:
                subitem = subitem.strip().lower()
                if len(subitem) >= 3:
                    # Clean up the item
                    subitem = re.sub(r'\s+(that|which|may|can|often|sometimes|usually)\s+.*', '', subitem)
                    subitem = re.sub(r'\s*\([^)]*\)\s*', '', subitem)
                    subitem = subitem.strip()
                    
                    if subitem:
                        items.append(subitem)
    
    return items

def parse_threshold_values(text):
    """Parse threshold values with their test names and criteria"""
    thresholds = []
    
    # Handle N/A case
    if 'N/A' in text or text.strip() == '-N/A':
        return [{'test_name': 'N/A', 'threshold': 'N/A', 'full_text': 'N/A'}]
    
    # Split by line breaks or bullet points
    lines = re.split(r'[\n-]\s*', text)
    
    for line in lines:
        line = line.strip()
        if not line or len(line) < 5:
            continue
        
        # Parse different threshold formats
        # Format 1: "Test Name: threshold value"
        colon_match = re.match(r'([^:]+):\s*(.+)', line)
        if colon_match:
            test_name = colon_match.group(1).strip()
            threshold = colon_match.group(2).strip()
            thresholds.append({
                'test_name': test_name,
                'threshold': threshold,
                'full_text': line
            })
        # Format 2: Just the threshold description
        else:
            thresholds.append({
                'test_name': 'General',
                'threshold': line,
                'full_text': line
            })
    
    return thresholds

def calculate_symptom_weight(symptom):
    """Calculate symptom weight based on severity keywords"""
    weight = 1.0
    if any(keyword in symptom for keyword in ['severe', 'chronic', 'persistent', 'blood', 'fever','recent']):
        weight = 2.0
    elif any(keyword in symptom for keyword in ['mild', 'occasional', 'sometimes']):
        weight = 0.5
    return weight

# --- Step 2: Create DataFrame from PDF ---
def create_dataframe_from_pdf(pdf_path):
    """Create dataframe and disease info dictionary from PDF content"""
    pdf_content = extract_pdf_content(pdf_path)
    diseases_data, disease_info = parse_disease_data_from_pdf(pdf_content)
    df = pd.DataFrame(diseases_data)
    return df, disease_info

# Global variable to store disease information
DISEASE_INFO = {}

# --- Step 3: RAG System Setup ---
def setup_rag_system_local(pdf_path):
    """Set up RAG system with local sentence-transformer embeddings"""
    try:
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # Use local embeddings (same model you're already using)
        from langchain_community.embeddings import SentenceTransformerEmbeddings
        embeddings = SentenceTransformerEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
        
        vectorstore = FAISS.from_documents(splits, embeddings)
        return vectorstore
        
    except Exception as e:
        print(f"Local RAG system setup failed: {e}")
        return None

# --- Step 4: Enhanced Query Processing with RAG ---
def enhance_query_with_rag(user_query, vectorstore, llm):
    """Enhance user query with relevant context from PDF"""
    if vectorstore is None:
        return ""
    
    try:
        # Retrieve relevant documents
        relevant_docs = vectorstore.similarity_search(user_query, k=3)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Create enhanced prompt
        enhanced_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="""
            Based on the following medical context from authoritative sources:
            
            {context}
            
            User Query: {query}
            
            Extract relevant medical symptoms mentioned in the query, considering the medical context provided.
            Return symptoms as a comma-separated list.
            """
        )
        
        chain = LLMChain(llm=llm, prompt=enhanced_prompt)
        return chain.run(query=user_query, context=context)
    except Exception as e:
        print(f"RAG enhancement failed: {e}")
        return ""

# --- Main Setup ---
# Set environment variables for Azure OpenAI
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

# Load PDF and create DataFrame (replacing CSV loading)
pdf_path = "flows.pdf"  # Your PDF file path

print("Loading PDF and extracting disease data...")
try:
    df, DISEASE_INFO = create_dataframe_from_pdf(pdf_path)
    print(f"✅ Successfully loaded {len(df)} symptom entries")
    print(f"✅ Extracted information for {len(DISEASE_INFO)} diseases")
except Exception as e:
    print(f"❌ Error loading PDF: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Setup RAG system
print("\nSetting up RAG system...")
try:
    vectorstore = setup_rag_system_local(pdf_path)
    print("✅ RAG system setup complete!")
except Exception as e:
    print(f"⚠️ RAG system setup failed: {e}")
    vectorstore = None

# --- Step 5: Set up LangChain with OpenAI ---
print("\nInitializing Azure OpenAI LLM...")
try:
    llm = AzureChatOpenAI(
        azure_deployment="medicalChatbot",  
        api_version="2024-06-01",  
        temperature=0.3,  # Increased for more conversational responses
        max_tokens=512,
        model='gpt-4o',
        max_retries=2
    )
    print("✅ LLM initialized successfully")
except Exception as e:
    print(f"❌ Error initializing LLM: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ENHANCED: Conversational symptom extraction prompts
symptom_extraction_prompt = PromptTemplate(
    input_variables=["user_query", "rag_context"],
    template="""
You are a friendly, experienced doctor having a conversation with a patient.

Medical Context: {rag_context}

Patient says: {user_query}

As a doctor, extract ONLY the medical symptoms the patient has mentioned. 
Be precise and use medical terminology when appropriate.
Return symptoms as a comma-separated list.
If no symptoms are mentioned, return "none".

Examples:
- "I have fever and my head hurts" -> "fever, headache"
- "Yes, I have joint pain" -> "joint pain"
- "I feel really tired lately" -> "fatigue"
- "No, I don't have that" -> "none"
"""
)

symptom_chain = LLMChain(llm=llm, prompt=symptom_extraction_prompt)

# ENHANCED: Conversational follow-up symptom extraction
followup_symptom_prompt = PromptTemplate(
    input_variables=["question", "answer", "symptoms", "top_diseases", "rag_context"],
    template="""
You are a friendly doctor analyzing a patient's response to your question.

Your Question: {question}
Patient's Answer: {answer}
Patient's Current Symptoms: {symptoms}
Suspected Conditions: {top_diseases}

Based on the patient's response, determine what NEW symptoms they've mentioned.

RULES:
1. If patient says "yes" to your question about a symptom, extract that symptom from your question
2. If patient says "no", return "none"
3. If patient mentions additional symptoms, extract those too
4. Be conversational and understanding

Examples:
- Your question: "Are you experiencing any joint pain?" Patient: "Yes" -> "joint pain"
- Your question: "Do you have muscle aches?" Patient: "Yes, especially in my legs" -> "muscle pain"
- Your question: "Any nausea?" Patient: "No" -> "none"
- Patient: "Actually, I also have been vomiting" -> "vomiting"

Return new symptoms as comma-separated list, or "none" if no new symptoms.
"""
)

followup_symptom_chain = LLMChain(llm=llm, prompt=followup_symptom_prompt)

# ENHANCED: Medical normalization with friendly tone
normalization_prompt = PromptTemplate(
    input_variables=["raw_symptoms"],
    template="""
You are a doctor standardizing symptoms for medical records.

Patient mentioned: {raw_symptoms}

Convert these to standard medical terms.

Return ONLY a comma-separated list of standardized medical terms.
"""
)
normalize_chain = LLMChain(llm=llm, prompt=normalization_prompt)

# NEW: LLM-based question generation with medical justification
question_generation_prompt = PromptTemplate(
    input_variables=["current_symptoms", "top_diseases", "available_symptoms", "conversation_context"],
    template="""
You are an experienced, friendly doctor conducting a diagnostic interview.

PATIENT'S CURRENT SYMPTOMS: {current_symptoms}
SUSPECTED CONDITIONS: {top_diseases}
AVAILABLE SYMPTOMS TO ASK ABOUT: {available_symptoms}
CONVERSATION CONTEXT: {conversation_context}

Your task:
1. Generate ONE thoughtful, conversational question about a symptom from the available list
2. Choose the most medically relevant symptom to ask about
3. Provide clear medical justification for why this question is important

Format your response as:
QUESTION: [Your conversational question]
SYMPTOM: [The exact symptom you're asking about]
JUSTIFICATION: [Medical reasoning in friendly, understandable terms]

Be warm, professional, and explain why this question helps with diagnosis.
Make the question natural and conversational, not robotic.

Examples:
QUESTION: I'd like to ask about your urination - have you noticed any blood in your urine recently?
SYMPTOM: hematuria
JUSTIFICATION: Blood in urine can be an important early sign of several conditions I'm considering based on your other symptoms. It helps me narrow down the diagnosis and determine the urgency of your condition.
"""
)

question_generation_chain = LLMChain(llm=llm, prompt=question_generation_prompt)

# NEW: Symptom validation against PDF
def validate_symptom_against_pdf(symptom, top_diseases, df_grouped):
    """Check if the symptom exists in PDF data for any of the top diseases"""
    for disease in top_diseases:
        disease_row = df_grouped[df_grouped['disease'] == disease]
        if not disease_row.empty:
            disease_symptoms = disease_row.iloc[0]['symptom_clean']
            if symptom in disease_symptoms:
                return True, disease
    return False, None

# Create a conversation memory for tracking the session
conversation_memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="user_input",
    output_key="assistant_response"
)

# --- Step 6: Embed symptoms using sentence transformer embedding ---
# Load embedding model
print("\nLoading embedding model...")
try:
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("✅ Embedding model loaded")
except Exception as e:
    print(f"❌ Error loading embedding model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Process symptoms
print("\nProcessing symptoms and creating embeddings...")
try:
    if len(df) == 0:
        print("❌ No data in dataframe!")
        exit(1)
        
    df["symptom_clean"] = df["symptom"].apply(lambda x: x.strip().lower())
    print(f"✅ Cleaned {len(df)} symptoms")
    
    # Create embeddings in batches to avoid memory issues
    print("Creating embeddings (this may take a moment)...")
    symptoms_list = df["symptom_clean"].tolist()
    embeddings = []
    batch_size = 100
    
    for i in range(0, len(symptoms_list), batch_size):
        batch = symptoms_list[i:i+batch_size]
        batch_embeddings = embedding_model.encode(batch, convert_to_numpy=True)
        embeddings.extend(batch_embeddings)
        print(f"   Processed {min(i+batch_size, len(symptoms_list))}/{len(symptoms_list)} symptoms...")
    
    df["symptom_embedding"] = embeddings
    print("✅ Created embeddings for all symptoms")
    
    # Group by disease
    print("\nGrouping symptoms by disease...")
    df_grouped = df.groupby("disease").agg({
        "symptom_clean": list,
        "symptom_embedding": list,
        "weight": list,
        "evidence_source": lambda x: ", ".join(set(x))
    }).reset_index()
    
    print(f"✅ Grouped into {len(df_grouped)} diseases")
    
except Exception as e:
    print(f"❌ Error processing symptoms: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# ENHANCED: Symptom tracking class with disease confidence tracking
class SymptomTracker:
    def __init__(self):
        self.positive_symptoms = set()
        self.negative_symptoms = set()
        self.asked_symptoms = set()
        self.disease_scores = {}
        self.matched_symptoms = {}
        self.disease_confidence = {}  # New: Track confidence per disease
        self.disease_negative_impact = {}  # New: Track negative symptom impact
        self.conversation_history = []
        self.question_count = 0
    
    def add_positive_symptom(self, symptom):
        """Add a symptom that the user has"""
        self.positive_symptoms.add(symptom)
        self.asked_symptoms.add(symptom)
        print(f"✓ Patient confirmed: {symptom}")
    
    def add_negative_symptom(self, symptom):
        """Add a symptom that the user doesn't have"""
        self.negative_symptoms.add(symptom)
        self.asked_symptoms.add(symptom)
        print(f"✗ Patient denied: {symptom}")
        
        # NEW: Apply negative impact to diseases that have this symptom
        self.apply_negative_symptom_impact(symptom)
    
    def apply_negative_symptom_impact(self, denied_symptom):
        """Reduce confidence for diseases that have the denied symptom"""
        global df_grouped
        
        print(f"📉 Analyzing impact of denied symptom: {denied_symptom}")
        
        for _, row in df_grouped.iterrows():
            disease = row["disease"]
            symptoms = row["symptom_clean"]
            weights = row["weight"]
            
            # Check if this disease has the denied symptom
            if denied_symptom in symptoms:
                symptom_index = symptoms.index(denied_symptom)
                symptom_weight = weights[symptom_index]
                
                # Calculate negative impact based on symptom weight
                negative_impact = symptom_weight * 0.8  # 80% of the original weight as penalty
                
                # Apply negative impact
                if disease not in self.disease_negative_impact:
                    self.disease_negative_impact[disease] = 0
                self.disease_negative_impact[disease] += negative_impact
                
                print(f"   📊 {disease}: -{negative_impact:.2f} confidence penalty")
    
    def has_been_asked(self, symptom):
        """Check if we've already asked about this symptom"""
        return symptom in self.asked_symptoms
    
    def get_all_positive_symptoms(self):
        """Get all symptoms the user has"""
        return list(self.positive_symptoms)
    
    def add_to_conversation(self, question, answer, justification):
        """Add to conversation history"""
        self.conversation_history.append({
            'question': question,
            'answer': answer,
            'justification': justification,
            'question_number': self.question_count
        })
        self.question_count += 1
    
    def calculate_disease_confidence(self, disease, base_score, matched_symptom_count):
        """Calculate confidence for a specific disease with softer scaling"""
        # Base confidence from matched symptom relevance
        base_confidence = 1 - np.exp(-0.15 * base_score)  # nonlinear smoothing (0 → 1 asymptotic)
        
        # Symptom count influence (logarithmic, diminishing returns)
        symptom_factor = min(np.log1p(matched_symptom_count) / 4.0, 0.5)
        
        # Negative symptom penalty (stronger than before)
        negative_impact = self.disease_negative_impact.get(disease, 0)
        negative_factor = min(0.15 * negative_impact, 0.6)
        
        # Combine with dampening
        raw_conf = base_confidence + symptom_factor - negative_factor
        
        # Apply overall soft cap (never 100%)
        confidence = min(raw_conf * 0.9, 0.95)  # scale and cap at 95%
        confidence = max(confidence, 0.0)
        
        return confidence, negative_impact

    
    def get_disease_confidence_details(self, disease, base_score, matched_symptom_count):
        """Get detailed confidence breakdown for a disease"""
        confidence, negative_impact = self.calculate_disease_confidence(disease, base_score, matched_symptom_count)
        
        return {
            'confidence': confidence,
            'base_score': base_score,
            'matched_symptoms': matched_symptom_count,
            'negative_impact': negative_impact,
            'denied_symptoms': len([s for s in self.negative_symptoms if self._symptom_belongs_to_disease(s, disease)])
        }
    
    def _symptom_belongs_to_disease(self, symptom, disease):
        """Check if a symptom belongs to a specific disease"""
        global df_grouped
        disease_row = df_grouped[df_grouped['disease'] == disease]
        if not disease_row.empty:
            return symptom in disease_row.iloc[0]['symptom_clean']
        return False

# NEW: Enhanced symptom tracker with test tracking
class EnhancedSymptomTracker(SymptomTracker):
    def __init__(self):
        super().__init__()
        self.test_results = {}  # {disease: {test_name: result}}
        self.diseases_ruled_out_by_tests = set()
        self.test_conversation_history = []
        self.risk_factors_checked = {}  # {disease: [risk_factors]}
    
    def add_test_result(self, disease, test_name, result, interpretation):
        """Track test results for each disease"""
        if disease not in self.test_results:
            self.test_results[disease] = {}
        self.test_results[disease][test_name] = {
            'result': result,
            'interpretation': interpretation
        }
        
    def rule_out_disease(self, disease, reason):
        """Rule out a disease based on test results"""
        self.diseases_ruled_out_by_tests.add(disease)
        print(f"❌ {disease} ruled out: {reason}")
        
    def add_test_conversation(self, disease, test_name, question, answer, interpretation):
        """Track test-related conversation"""
        self.test_conversation_history.append({
            'disease': disease,
            'test_name': test_name,
            'question': question,
            'answer': answer,
            'interpretation': interpretation
        })
    
    def add_risk_factor(self, disease, risk_factor):
        """Track risk factors for diseases"""
        if disease not in self.risk_factors_checked:
            self.risk_factors_checked[disease] = []
        self.risk_factors_checked[disease].append(risk_factor)

# --- Step 7: Enhanced embedding-based matching (same as before)
# def enhanced_diagnose(user_input_text, df_grouped, model, symptom_tracker, threshold_sentence=0.55):
#     """Enhanced diagnosis with better symptom matching and synonym handling"""
    
#     # Create symptom synonyms for better matching
#     symptom_synonyms = {
#         'hematuria': ['blood in urine', 'blood urine', 'bloody urine'],
#         'urinary frequency': ['frequent urination', 'urination frequency', 'urinating often'],
#         'urinary retention': ['difficulty urinating', 'trouble urinating', 'hard to urinate'],
#         'urgency': ['urinary urgency', 'urgent urination', 'need to urinate urgently'],
#         'back pain': ['lower back pain', 'backache', 'spine pain'],
#         'hip pain': ['hip discomfort', 'hip ache'],
#         'weight loss': ['losing weight', 'unintentional weight loss'],
#         'bone pain': ['bone ache', 'skeletal pain'],
#         'erectile dysfunction': ['ed', 'impotence', 'sexual dysfunction'],
#         'fatigue': ['tiredness', 'exhaustion', 'tired'],
#         'pelvic pain': ['pelvic discomfort', 'pelvis pain'],
#         'joint pain': ['joint ache', 'arthritis pain', 'joint discomfort'],
#         'muscle pain': ['muscle ache', 'myalgia', 'muscle soreness']
#     }
    
#     combined_text = f"{user_input_text}".lower()
    
#     # Expand input with synonyms
#     expanded_input = combined_text
#     for standard_term, synonyms in symptom_synonyms.items():
#         for synonym in synonyms:
#             if synonym in expanded_input:
#                 expanded_input += f" {standard_term}"
    
#     sentence_embedding = model.encode(expanded_input, convert_to_numpy=True)
#     disease_scores = symptom_tracker.disease_scores
#     matched_symptoms = symptom_tracker.matched_symptoms

#     for _, row in df_grouped.iterrows():
#         disease = row["disease"]
#         symptoms = row["symptom_clean"]
#         embeddings = row["symptom_embedding"]
#         weights = row["weight"]

#         new_matched = []
#         score_increment = 0

#         for sym, emb, wt in zip(symptoms, embeddings, weights):
#             # Skip if this symptom is in negative symptoms
#             if sym in symptom_tracker.negative_symptoms:
#                 continue
                
#             sim_score = cosine_similarity([emb], [sentence_embedding])[0][0]
            
#             # Also check for exact matches and synonyms
#             exact_match = False
#             for standard_term, synonyms in symptom_synonyms.items():
#                 if (sym == standard_term or sym in synonyms) and (standard_term in expanded_input or any(syn in expanded_input for syn in synonyms)):
#                     exact_match = True
#                     break
            
#             if sim_score >= threshold_sentence or exact_match:
#                 if disease not in matched_symptoms or sym not in matched_symptoms[disease]:
#                     new_matched.append(sym)
#                     match_weight = wt * 1.0 if exact_match else wt * 0.6
#                     score_increment += match_weight

#         if new_matched:
#             disease_scores[disease] = disease_scores.get(disease, 0) + score_increment
#             matched_symptoms[disease] = list(set(matched_symptoms.get(disease, []) + new_matched))

#     symptom_tracker.disease_scores = disease_scores
#     symptom_tracker.matched_symptoms = matched_symptoms

#     ranked = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)
#     return ranked, matched_symptoms


def enhanced_diagnose(user_input_text, df_grouped, model, symptom_tracker, threshold_sentence=0.55):
    """
    HYBRID diagnosis:
    Combines local (symptom-level) similarity with global (contextual) similarity.
    """
    disease_scores = {}
    matched_symptoms = {}

    # 1) Local (per-symptom) similarity scoring
    for _, row in df_grouped.iterrows():
        disease = row["disease"]
        # pull the INNER lists/arrays directly
        disease_symptom_embs = np.vstack(row["symptom_embedding"])            # shape: (n_symptoms, dim)
        disease_symptom_texts = row["symptom_clean"]                           # list[str], aligned with embeddings

        total_score = 0.0

        for symptom_text in symptom_tracker.positive_symptoms:                 # only positives
            # encode once per user symptom
            symptom_emb = model.encode(symptom_text, convert_to_numpy=True).reshape(1, -1)

            # cosine to all disease symptoms
            sim_scores = cosine_similarity(symptom_emb, disease_symptom_embs)[0]
            best_idx = int(np.argmax(sim_scores))
            best_sim = float(sim_scores[best_idx])

            if best_sim >= threshold_sentence:
                # append the **disease-side** canonical symptom, not the patient's phrasing
                matched_symptoms.setdefault(disease, [])
                if disease_symptom_texts[best_idx] not in matched_symptoms[disease]:
                    matched_symptoms[disease].append(disease_symptom_texts[best_idx])
                total_score += best_sim

        disease_scores[disease] = total_score

    # 2) Global (patient summary vs. disease mean)
    if symptom_tracker.positive_symptoms:
        combined_text = " ".join(sorted(symptom_tracker.positive_symptoms))
        patient_emb = model.encode(combined_text, convert_to_numpy=True).reshape(1, -1)

        for _, row in df_grouped.iterrows():
            disease = row["disease"]
            disease_embs = np.vstack(row["symptom_embedding"])                 # (n_symptoms, dim)
            disease_mean_emb = np.mean(disease_embs, axis=0, keepdims=True)    # (1, dim)
            global_sim = float(cosine_similarity(patient_emb, disease_mean_emb)[0][0])

            local_score = disease_scores.get(disease, 0.0)
            # keep scales comparable: both components in [0, +)
            disease_scores[disease] = 0.7 * global_sim + 0.3 * local_score

    # 3) *** CRITICAL ***: update the tracker so downstream confidence uses fresh state
    symptom_tracker.disease_scores = disease_scores
    symptom_tracker.matched_symptoms = matched_symptoms

    # 4) Return ranked
    ranked_diseases = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_diseases, matched_symptoms



# NEW: LLM-based intelligent question generation
def generate_llm_question(symptom_tracker, df_grouped):
    """Generate question using LLM with medical justification"""
    
    # Get current context
    current_symptoms = symptom_tracker.get_all_positive_symptoms()
    top_diseases = [disease for disease, _ in sorted(symptom_tracker.disease_scores.items(), key=lambda x: x[1], reverse=True)[:3]]
    
    # Get available symptoms from top diseases that haven't been asked
    available_symptoms = set()
    for disease in top_diseases:
        disease_row = df_grouped[df_grouped['disease'] == disease]
        if not disease_row.empty:
            disease_symptoms = disease_row.iloc[0]['symptom_clean']
            for symptom in disease_symptoms:
                if not symptom_tracker.has_been_asked(symptom):
                    available_symptoms.add(symptom)
    
    # If no top diseases yet, get from all diseases
    if not available_symptoms:
        for _, row in df_grouped.iterrows():
            for symptom in row['symptom_clean']:
                if not symptom_tracker.has_been_asked(symptom):
                    available_symptoms.add(symptom)
    
    if not available_symptoms:
        return None, None, None
    
    # Prepare conversation context
    conversation_context = "This is the beginning of our consultation."
    if symptom_tracker.conversation_history:
        recent_history = symptom_tracker.conversation_history[-2:]  # Last 2 exchanges
        conversation_context = " ".join([f"Previous: {h['question']} -> {h['answer']}" for h in recent_history])
    
    # Generate question using LLM
    try:
        response = question_generation_chain.run(
            current_symptoms=", ".join(current_symptoms) if current_symptoms else "None reported yet",
            top_diseases=", ".join(top_diseases) if top_diseases else "Gathering initial information",
            available_symptoms=", ".join(list(available_symptoms)[:10]),  # Limit to 10 symptoms
            conversation_context=conversation_context
        )
        
        # Parse the response
        lines = response.strip().split('\n')
        question = None
        symptom = None
        justification = None
        
        for line in lines:
            if line.startswith('QUESTION:'):
                question = line.replace('QUESTION:', '').strip()
            elif line.startswith('SYMPTOM:'):
                symptom = line.replace('SYMPTOM:', '').strip()
            elif line.startswith('JUSTIFICATION:'):
                justification = line.replace('JUSTIFICATION:', '').strip()
        
        # Validate that the symptom exists in our PDF data
        if symptom and top_diseases:
            is_valid, disease_found = validate_symptom_against_pdf(symptom, top_diseases, df_grouped)
            if not is_valid:
                # Try with all diseases if not found in top diseases
                all_diseases = df_grouped['disease'].tolist()
                is_valid, disease_found = validate_symptom_against_pdf(symptom, all_diseases, df_grouped)
                
            # if not is_valid:
            #     print(f"⚠️  Generated symptom '{symptom}' not found in PDF data. Skipping...")
            #     return None, None, None
        
        return question, symptom, justification
        
    except Exception as e:
        print(f"Error generating LLM question: {e}")
        return None, None, None

# NEW: Test question generation using threshold values
test_question_generation_prompt = PromptTemplate(
    input_variables=["disease", "test_name", "threshold", "test_type"],
    template="""
You are a friendly doctor asking about medical test results.

Disease being evaluated: {disease}
Test: {test_name}
Threshold for diagnosis: {threshold}
Test type: {test_type}

Generate a warm, conversational question asking if the patient has had this test done recently.
Make it natural and easy to understand. If it's a complex test, briefly explain what it is.

Format your response as:
QUESTION: [Your conversational question]
EXPLANATION: [Brief, friendly explanation if needed]

Examples:
- For CBC/Blood count: "Have you had a recent blood test done? If so, do you remember what your platelet count or hemoglobin level was?"
- For specialized tests: "Have you had a chest X-ray recently? It's an imaging test that helps us see your lungs."
- For symptoms as tests: "Have you been experiencing this symptom for more than 2 weeks?"
"""
)

test_question_chain = LLMChain(llm=llm, prompt=test_question_generation_prompt)

# NEW: Test result interpretation using threshold values
test_interpretation_prompt = PromptTemplate(
    input_variables=["disease", "test_name", "threshold", "patient_response"],
    template="""
You are a friendly doctor interpreting a patient's test results.

Disease being evaluated: {disease}
Test: {test_name}
Threshold for diagnosis: {threshold}
Patient's response: {patient_response}

Analyze if the patient's response meets the threshold criteria.

Return your analysis in this format:
STATUS: [MEETS_THRESHOLD/BELOW_THRESHOLD/UNCLEAR/NOT_DONE]
EXPLANATION: [Brief, friendly explanation]

Guidelines:
- MEETS_THRESHOLD: Patient's value meets the criteria for this disease
- BELOW_THRESHOLD: Patient's value rules out this disease
- UNCLEAR: Cannot determine from the response
- NOT_DONE: Patient hasn't had this test

Examples:
- Threshold: "< 150,000 platelets", Patient: "120,000" → MEETS_THRESHOLD
- Threshold: "≥ 4 ng/mL", Patient: "2.5" → BELOW_THRESHOLD
- Patient: "I don't remember" → UNCLEAR
- Patient: "No, I haven't had that test" → NOT_DONE
"""
)

test_interpretation_chain = LLMChain(llm=llm, prompt=test_interpretation_prompt)

# NEW: Phase 2 - Test threshold evaluation using disease info
def evaluate_test_thresholds_phase2(symptom_tracker, ranked_diseases):
    """Phase 2: Ask about diagnostic test results for top diseases using extracted threshold values"""
    
    print("\n" + "="*70)
    print("📋 PHASE 2: DIAGNOSTIC TEST EVALUATION")
    print("="*70)
    print("\n👨‍⚕️ Doctor: Now I'd like to ask about any recent medical tests you may have had.")
    print("This will help me confirm or rule out certain conditions.\n")
    
    # Focus on top 3-5 diseases
    diseases_to_evaluate = [disease for disease, _ in ranked_diseases[:3]]
    final_candidates = []
    
    for disease in diseases_to_evaluate:
        if disease in symptom_tracker.diseases_ruled_out_by_tests:
            continue
        
        # Get threshold values for this disease from DISEASE_INFO
        if disease not in DISEASE_INFO:
            print(f"⚠️  No disease information found for {disease}")
            final_candidates.append(disease)
            continue
        
        disease_data = DISEASE_INFO[disease]
        threshold_values = disease_data.get('threshold_values', [])
        
        print(f"\n🔬 Evaluating: {disease}")
        
        # Check if this disease has threshold values
        if not threshold_values or (len(threshold_values) == 1 and threshold_values[0]['test_name'] == 'N/A'):
            print(f"   ℹ️  {disease} doesn't require specific diagnostic tests")
            final_candidates.append(disease)
            continue
        
        # Ask about each threshold test
        disease_still_possible = True
        tests_total = len(threshold_values)
        tests_negative = 0
        
        for threshold in threshold_values:
            test_name = threshold['test_name']
            threshold_value = threshold['threshold']
            full_text = threshold['full_text']
            
            # Skip if it's a general threshold or N/A
            if test_name == 'N/A' or test_name == 'General':
                continue
            
            # Determine test type
            test_type = "laboratory test"
            if any(word in test_name.lower() for word in ['x-ray', 'ct', 'mri', 'ultrasound', 'imaging']):
                test_type = "imaging test"
            elif any(word in test_name.lower() for word in ['spirometry', 'ekg', 'ecg']):
                test_type = "diagnostic procedure"
            
            # Generate conversational question
            try:
                question_response = test_question_chain.run(
                    disease=disease,
                    test_name=test_name,
                    threshold=threshold_value,
                    test_type=test_type
                )
                
                # Parse question and explanation
                lines = question_response.strip().split('\n')
                question = ""
                explanation = ""
                
                for line in lines:
                    if line.startswith('QUESTION:'):
                        question = line.replace('QUESTION:', '').strip()
                    elif line.startswith('EXPLANATION:'):
                        explanation = line.replace('EXPLANATION:', '').strip()
                
                if explanation:
                    print(f"\n💡 {explanation}")
                
                print(f"\n👨‍⚕️ Doctor: {question}")
                patient_response = input("🗣️  Your response: ")
                
                # Interpret the response
                interpretation_result = test_interpretation_chain.run(
                    disease=disease,
                    test_name=test_name,
                    threshold=threshold_value,
                    patient_response=patient_response
                )
                
                # Parse interpretation
                status = None
                explanation = ""
                
                for line in interpretation_result.strip().split('\n'):
                    if line.startswith('STATUS:'):
                        status = line.replace('STATUS:', '').strip()
                    elif line.startswith('EXPLANATION:'):
                        explanation = line.replace('EXPLANATION:', '').strip()
                
                print(f"   💡 {explanation}")
                
                # Track the conversation
                symptom_tracker.add_test_conversation(
                    disease, test_name, question, patient_response, explanation
                )

                
                
                # Process the result
                if status == "BELOW_THRESHOLD":
                    tests_negative += 1
                    symptom_tracker.add_test_result(disease, test_name, patient_response, "negative")
                    print(f"   ❌ Test result does not support {disease}")
                    if tests_negative == tests_total:
                        symptom_tracker.rule_out_disease(disease, explanation)
                        disease_still_possible = False
                        break
                elif status == "MEETS_THRESHOLD":
                    symptom_tracker.add_test_result(disease, test_name, patient_response, "positive")
                    print(f"   ✅ Test result supports {disease} diagnosis")
                elif status == "NOT_DONE":
                    print(f"   ℹ️  Test not performed - continuing evaluation")
                else:  # UNCLEAR
                    print(f"   ❓ Unable to interpret test result clearly")
                    
            except Exception as e:
                print(f"Error processing test for {disease}: {e}")
                continue
        
        if disease_still_possible:
            final_candidates.append(disease)
    
    return final_candidates

# NEW: Risk factor evaluation
risk_factor_prompt = PromptTemplate(
    input_variables=["disease", "risk_factors", "patient_info"],
    template="""
You are a doctor evaluating risk factors for a disease.

Disease: {disease}
Risk factors for this disease: {risk_factors}
Patient information collected so far: {patient_info}

Generate 1-2 important questions about risk factors that haven't been addressed yet.
Make questions conversational and non-judgmental.

Format:
QUESTIONS: [Your questions, separated by |]
FACTORS: [The risk factors being asked about, separated by |]

Example:
QUESTIONS: Can you tell me about your family medical history, particularly any cases of diabetes or heart disease?|Do you smoke or have you smoked in the past?
FACTORS: family history|smoking
"""
)

risk_factor_chain = LLMChain(llm=llm, prompt=risk_factor_prompt)

# Rest of the utility functions remain the same...
# ENHANCED: Calculate confidence with negative symptom impact
def calculate_confidence(symptom_tracker):
    """Calculate diagnostic confidence based on multiple factors including negative symptoms"""
    disease_scores = symptom_tracker.disease_scores
    matched_symptoms = symptom_tracker.matched_symptoms
    
    if not disease_scores:
        return 0.0
    
    top_disease, top_score = max(disease_scores.items(), key=lambda x: x[1])
    
    # Factor 1: Number of symptoms matched
    symptom_count = len(matched_symptoms.get(top_disease, []))
    
    # Factor 2: Score margin over second best
    sorted_scores = sorted(disease_scores.values(), reverse=True)
    if len(sorted_scores) > 1 and sorted_scores[1] > 0:
        margin = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]
    else:
        margin = 1.0
    
    # Factor 3: Negative symptom impact
    negative_impact = symptom_tracker.disease_negative_impact.get(top_disease, 0)
    negative_penalty = min(negative_impact * 0.05, 0.5)  # Cap penalty at 50%
    
    # Enhanced confidence calculation with negative symptom consideration
    confidence = (symptom_count * 0.10) + (margin * 0.25) + (top_score * 0.12) - negative_penalty
    confidence = np.tanh(confidence) * 0.9  # compress and scale to 0–0.9

    return confidence

def display_detailed_assessment(symptom_tracker, ranked_diseases, matched_symptoms):
    """Display detailed assessment with confidence breakdown for each disease"""
    print(f"\n📊 DETAILED DISEASE ASSESSMENT:")
    print("=" * 70)
    
    if not ranked_diseases:
        print("   No diseases identified yet. Continuing assessment...")
        return
    
    overall_confidence = calculate_confidence(symptom_tracker)
    print(f"🎯 Overall Diagnostic Confidence: {overall_confidence:.1%}")
    
    print(f"\n🔍 Top Conditions Analysis:")
    print("-" * 50)
    
    for i, (disease, base_score) in enumerate(ranked_diseases[:3], 1):
        symptoms_matched = matched_symptoms.get(disease, [])
        confidence_details = symptom_tracker.get_disease_confidence_details(
            disease, base_score, len(symptoms_matched)
        )
        
        print(f"\n{i}. {disease}")
        print(f"   📈 Base Score: {base_score:.2f}")
        print(f"   🎯 Confidence: {confidence_details['confidence']:.1%}")
        print(f"   ✅ Similar Symptoms ({len(symptoms_matched)}): {', '.join(symptoms_matched) if symptoms_matched else 'None'}")
        
        # Show denied symptoms that affect this disease
        denied_for_disease = [s for s in symptom_tracker.negative_symptoms 
                            if symptom_tracker._symptom_belongs_to_disease(s, disease)]
        if denied_for_disease:
            print(f"   ❌ Denied Symptoms ({len(denied_for_disease)}): {', '.join(denied_for_disease)}")
            print(f"   📉 Negative Impact: -{confidence_details['negative_impact']:.2f}")
        else:
            print(f"   ❌ Denied Symptoms: None")
        
        print(f"   🧮 Weight Calculation:")
        print(f"      • Matched symptoms bonus: +{len(symptoms_matched) * 0.15:.2f}")
        print(f"      • Base score factor: +{base_score * 0.1:.2f}")
        if confidence_details['negative_impact'] > 0:
            print(f"      • Denial penalty: -{confidence_details['negative_impact'] * 0.1:.2f}")
        print(f"      • Final confidence: {confidence_details['confidence']:.1%}")

def should_continue_diagnosis(symptom_tracker, question_count, min_questions=10, min_symptoms=4):
    """Enhanced decision making including confidence considerations""" 
    
    # Always ask minimum questions
    if question_count < min_questions:
        return True
    
    # Check if we have enough symptoms
    disease_scores = symptom_tracker.disease_scores
    matched_symptoms = symptom_tracker.matched_symptoms
    
    if not disease_scores:
        return True
    
    max_symptoms = max(len(symptoms) for symptoms in matched_symptoms.values()) if matched_symptoms else 0
    if max_symptoms < min_symptoms:
        return True
    
    # Check overall confidence
    confidence = calculate_confidence(symptom_tracker)
    if confidence < 0.70:  
        return True
    
    # Check if top disease has good confidence
    top_disease = max(disease_scores.items(), key=lambda x: x[1])[0]
    top_disease_confidence = symptom_tracker.calculate_disease_confidence(
        top_disease, 
        disease_scores[top_disease], 
        len(matched_symptoms.get(top_disease, []))
    )[0]
    
    if top_disease_confidence < 0.7:
        return True
    
    # Check if top diseases are too close in confidence
    if len(disease_scores) > 1:
        sorted_diseases = sorted(disease_scores.items(), key=lambda x: x[1], reverse=True)
        top_conf = symptom_tracker.calculate_disease_confidence(
            sorted_diseases[0][0], sorted_diseases[0][1], 
            len(matched_symptoms.get(sorted_diseases[0][0], []))
        )[0]
        second_conf = symptom_tracker.calculate_disease_confidence(
            sorted_diseases[1][0], sorted_diseases[1][1], 
            len(matched_symptoms.get(sorted_diseases[1][0], []))
        )[0]
        
        # if second_conf > 0 and top_conf / second_conf < 1.5:
        #     return True
    
    # Stop if we've asked too many questions
    if question_count >= 20:
        return False
    
    return False

def normalize_symptoms(symptoms_list):
    """Normalize and deduplicate symptoms"""
    normalized = []
    
    # Standardize terms
    term_mapping = {
        'blood in urine': 'hematuria',
        'difficulty urinating': 'urinary retention',
        'frequent urination': 'urinary frequency',
        'urinary urgency': 'urgency',
        'lower back pain': 'back pain',
        'unintentional weight loss': 'weight loss',
        'tiredness': 'fatigue',
        'pelvic discomfort': 'pelvic pain',
    }
    
    for symptom in symptoms_list:
        symptom_lower = symptom.lower().strip()
        # Apply mapping
        mapped_symptom = term_mapping.get(symptom_lower, symptom_lower)
        if mapped_symptom not in normalized:
            normalized.append(mapped_symptom)
    
    return normalized

# NEW: Generate final diagnosis with test results
final_diagnosis_prompt = PromptTemplate(
    input_variables=["symptoms", "initial_diseases", "test_results", "ruled_out_diseases", "disease_info"],
    template="""
You are an experienced doctor providing a final diagnosis.

Patient's symptoms: {symptoms}
Initial suspected conditions: {initial_diseases}
Test results collected: {test_results}
Conditions ruled out by tests: {ruled_out_diseases}
Disease information: {disease_info}

Based on both symptoms and test results, provide:
1. The most likely diagnosis with confidence level
2. Explanation of why this diagnosis fits best
3. Relevant risk factors to consider
4. Recommended next steps
5. When to seek immediate care

Be warm, professional, and thorough.
"""
)

final_diagnosis_chain = LLMChain(llm=llm, prompt=final_diagnosis_prompt)

# MODIFIED: Enhanced multi-turn session with Phase 2
def enhanced_multi_turn_session_with_tests():
    # Initialize enhanced symptom tracker
    symptom_tracker = EnhancedSymptomTracker()
    
    # Phase 1: Symptom collection (your existing code)
    last_question = None
    last_symptom_asked = None
    last_justification = None
    question_count = 0
    max_questions = 20

    print("🏥 ENHANCED MEDICAL DIAGNOSTIC SYSTEM WITH TEST EVALUATION")
    print("=" * 70)
    print("Hello! I'm your AI medical assistant. I'll help you understand")
    print("your symptoms and may ask about any recent medical tests.")
    print("=" * 70)

    # Phase 1: Complete symptom collection first
    phase1_complete = False
    
    while not phase1_complete:
        if question_count == 0:
            user_query = input(f"\n👨‍⚕️ Doctor: Can you tell me about the symptoms you're experiencing? ")
        else:
            user_query = input(f"\n🗣️  Your response: ")
            
        if user_query.lower() == 'exit':
            print("\n👨‍⚕️ Doctor: Thank you for your time. Please consult with a healthcare professional.")
            return

        # RAG context lookup
        rag_context = "No additional context available"
        if vectorstore:
            try:
                docs = vectorstore.similarity_search(user_query, k=2)
                rag_context = "\n".join(d.page_content for d in docs)
            except Exception:
                pass

        # Process symptoms (existing code)
        if last_question and last_symptom_asked:
            top_names = [d for d, _ in symptom_tracker.disease_scores.items()][:3] if symptom_tracker.disease_scores else []
            extracted_symptoms = followup_symptom_chain.run(
                question=last_question,
                answer=user_query,
                symptoms=", ".join(symptom_tracker.get_all_positive_symptoms()),
                top_diseases=", ".join(top_names),
                rag_context=rag_context
            )
            
            user_answer_lower = user_query.lower().strip()
            if user_answer_lower in ['yes', 'y', 'yeah', 'yep', 'sure', 'definitely']:
                symptom_tracker.add_positive_symptom(last_symptom_asked)
                new_symptoms = [last_symptom_asked]
            elif user_answer_lower in ['no', 'n', 'nope', 'not really', 'nah']:
                symptom_tracker.add_negative_symptom(last_symptom_asked)
                new_symptoms = []
            else:
                if extracted_symptoms.strip().lower() in ["none", "no symptoms", ""]:
                    new_symptoms = []
                else:
                    normalized_output = normalize_chain.run(raw_symptoms=extracted_symptoms)
                    new_symptoms = [
                        s.strip().lower()
                        for s in re.split(r"[,\n]+", normalized_output)
                        if s.strip()
                    ]
                    new_symptoms = normalize_symptoms(new_symptoms)
                    
                    for symptom in new_symptoms:
                        symptom_tracker.add_positive_symptom(symptom)
            
            symptom_tracker.add_to_conversation(last_question, user_query, last_justification)
        else:
            # Initial symptom extraction
            extracted_symptoms = symptom_chain.run(
                user_query=user_query,
                rag_context=rag_context
            )
            
            if extracted_symptoms.strip().lower() in ["none", "no symptoms", ""]:
                new_symptoms = []
            else:
                normalized_output = normalize_chain.run(raw_symptoms=extracted_symptoms)
                new_symptoms = [
                    s.strip().lower()
                    for s in re.split(r"[,\n]+", extracted_symptoms + "," + normalized_output)
                    if s.strip()
                ]
                new_symptoms = normalize_symptoms(new_symptoms)
                
                for symptom in new_symptoms:
                    symptom_tracker.add_positive_symptom(symptom)

        # Run diagnosis
        all_text = " ".join(new_symptoms) if new_symptoms else user_query
        ranked_diseases, matched_symptoms = enhanced_diagnose(
            all_text, df_grouped, embedding_model, symptom_tracker
        )
        
        question_count += 1
        
        # Display current assessment
        display_detailed_assessment(symptom_tracker, ranked_diseases, matched_symptoms)
        
        # Decide whether to continue Phase 1
        if should_continue_diagnosis(symptom_tracker, question_count) and question_count < max_questions:
            print(f"\n🤔 Let me think about what to ask next...")
            next_question, next_symptom, justification = generate_llm_question(symptom_tracker, df_grouped)
            
            if next_question and next_symptom and justification:
                print(f"\n💡 Medical Reasoning:")
                print(f"   {justification}")
                print(f"\n👨‍⚕️ Doctor: {next_question}")
                
                last_question = next_question
                last_symptom_asked = next_symptom
                last_justification = justification
                continue
            else:
                print("\n👨‍⚕️ Doctor: I think I have enough information about your symptoms.")

                phase1_complete = True
        else:
            phase1_complete = True
    
    # Phase 1 is complete - Show top 3 diseases before moving to Phase 2
    print("\n" + "="*70)
    print("📊 PHASE 1 COMPLETE - INITIAL DIAGNOSIS")
    print("="*70)

    try:
        print("\n📊 Visualizing Symptom-to-Disease Influence Map...")
        visualize_disease_symptom_attribution(symptom_tracker, df_grouped, embedding_model)
    except Exception as e:
        print(f"⚠️ Failed to visualize attribution map: {e}")
    
    if ranked_diseases:
        print("\n👨‍⚕️ Doctor: Based on your symptoms, here are the most likely conditions:")
        print("\n🏥 TOP 3 POSSIBLE CONDITIONS:")
        print("-" * 50)
        
        for i, (disease, score) in enumerate(ranked_diseases[:3], 1):
            symptoms_matched = matched_symptoms.get(disease, [])
            confidence_details = symptom_tracker.get_disease_confidence_details(
                disease, score, len(symptoms_matched)
            )
            
            print(f"\n{i}. {disease}")
            print(f"   📊 Likelihood Score: {score:.2f}")
            print(f"   🎯 Confidence: {confidence_details['confidence']:.1%}")
            print(f"   ✅ Matching Symptoms: {', '.join(symptoms_matched) if symptoms_matched else 'None'}")
            
            # Show brief disease info if available
            if disease in DISEASE_INFO:
                risk_factors = DISEASE_INFO[disease].get('risk_factors', [])
                if risk_factors:
                    print(f"   ⚠️  Common Risk Factors: {', '.join(risk_factors[:3])}")
    else:
        print("\n👨‍⚕️ Doctor: I need more information to determine possible conditions.")
    print(f"   • Symptoms Confirmed: {symptom_tracker.get_all_positive_symptoms()}")
    
    # Ask if patient wants to proceed to test evaluation
    print("\n" + "-"*70)
    proceed = input("\n👨‍⚕️ Doctor: To confirm or rule out these conditions, I'd like to ask about any recent medical tests you may have had. Would you like to continue? (yes/no): ")
    
    if proceed.lower() not in ['yes', 'y', 'yeah', 'yep', 'sure']:
        print("\n👨‍⚕️ Doctor: No problem! Please consult with a healthcare professional for proper diagnosis.")
        print("Based on our conversation, the most likely conditions are:")
        for i, (disease, _) in enumerate(ranked_diseases[:3], 1):
            print(f"{i}. {disease}")
        return
    
    # Now start Phase 2: Test threshold evaluation
    print("\n" + "="*70)
    print("📋 PHASE 2: DIAGNOSTIC TEST EVALUATION")
    print("="*70)
    print("\n👨‍⚕️ Doctor: Great! Let me ask about specific tests that can help confirm or rule out these conditions.")
    
    # Phase 2: Test threshold evaluation
    final_candidates = evaluate_test_thresholds_phase2(symptom_tracker, ranked_diseases)
    print(final_candidates)
    # # Generate final diagnosis
    print("\n" + "="*70)
    print("📋 FINAL DIAGNOSIS AFTER TEST EVALUATION")
    print("="*70)
    
    if final_candidates:
        # Prepare disease info for final diagnosis
        disease_info_str = json.dumps({
            disease: DISEASE_INFO.get(disease, {})
            for disease in final_candidates[:3]
        }, indent=2)
        
        test_results_summary = json.dumps(symptom_tracker.test_results, indent=2)
        ruled_out_summary = list(symptom_tracker.diseases_ruled_out_by_tests)
        
        final_diagnosis = final_diagnosis_chain.run(
            symptoms=", ".join(symptom_tracker.get_all_positive_symptoms()),
            initial_diseases=", ".join([d for d, _ in ranked_diseases[:3]]),
            test_results=test_results_summary,
            ruled_out_diseases=", ".join(ruled_out_summary) if ruled_out_summary else "None",
            disease_info=disease_info_str
        )
        
        print(f"\n👨‍⚕️ Doctor: {final_diagnosis}")
        
        # Show the most likely diagnosis
        most_likely = final_candidates[0]
        print(f"\n🎯 Most Likely Diagnosis: {most_likely}")
        
        # Show supporting evidence
        print(f"\n📊 Supporting Evidence:")
        print(f"   • Matching symptoms: {', '.join(matched_symptoms.get(most_likely, []))}")
        if most_likely in symptom_tracker.test_results:
            print(f"   • Supporting test results:")
            for test_name, result in symptom_tracker.test_results[most_likely].items():
                print(f"     - {test_name}: {result['result']}")
        
        # Show risk factors if available
        if most_likely in DISEASE_INFO:
            risk_factors = DISEASE_INFO[most_likely].get('risk_factors', [])
            if risk_factors:
                print(f"\n⚠️  Risk Factors to Consider:")
                for rf in risk_factors[:5]:  # Show top 5
                    print(f"   • {rf}")
    else:
        print("\n👨‍⚕️ Doctor: Based on your symptoms and test results, I couldn't identify")
        print("a specific condition with high confidence. This could mean:")
        print("• Your symptoms may be related to a condition not in my knowledge base")
        print("• Additional tests may be needed for a definitive diagnosis")
        print("• Your symptoms might be related to multiple conditions")
    
    # Enhanced report generation
    print(f"\n📝 Consultation Summary:")
    print("=" * 50)
    print(f"   • Phase 1 - Symptom Questions: {question_count}")
    print(f"   • Symptoms Confirmed: {len(symptom_tracker.get_all_positive_symptoms())}")
    print(f"   • Symptoms Denied: {len(symptom_tracker.negative_symptoms)}")
    print(f"   • Initial Top 3 Diagnoses: {', '.join([d for d, _ in ranked_diseases[:3]])}")
    print(f"   • Phase 2 - Tests Discussed: {len(symptom_tracker.test_conversation_history)}")
    print(f"   • Diseases Ruled Out by Tests: {len(symptom_tracker.diseases_ruled_out_by_tests)}")
    
    if symptom_tracker.diseases_ruled_out_by_tests:
        print(f"\n❌ Conditions Ruled Out by Test Results:")
        for disease in symptom_tracker.diseases_ruled_out_by_tests:
            print(f"   • {disease}")

SYSTEM_READY = True

# --- Main execution ---
if __name__ == "__main__":
    try:
        print("🏥 ENHANCED CONVERSATIONAL MEDICAL DIAGNOSTIC SYSTEM")
        print("=" * 70)
        print("🤖 Features:")
        print("• LLM-powered conversational question generation")
        print("• Medical justification for each question")
        print("• PDF-based symptom validation")
        print("• Friendly, doctor-like interaction")
        print("• Comprehensive consultation summaries")
        print("=" * 70)
        print("\n⚠️  SAFETY NOTICE & MEDICAL DISCLAIMER:")
        print("This AI assessment is for educational purposes only.")
        print("Always consult with qualified healthcare professionals for:")
        print("• Proper medical diagnosis and treatment")
        print("• Interpretation of test results")
        print("• Medical decisions")
        
        
        # Check if all systems are ready
        if len(df) == 0:
            print("❌ Error: No medical data loaded from PDF")
            exit(1)
        
        print("\n🚀 System ready! Starting consultation...")
        
        # Start the enhanced conversational diagnostic session
        enhanced_multi_turn_session_with_tests()
        
    except KeyboardInterrupt:
        print("\n\n👋 Doctor: Consultation interrupted. Take care!")
    except Exception as e:
        print(f"\n❌ System error: {e}")
        print("Please check your configuration and try again.")
    finally:
        print("\n🏥 Thank you for using the Medical Diagnostic Assistant!")
        print("Remember: Always consult with healthcare professionals for medical advice.")
        print("\n© 2025 Maliha Tabassum. All rights reserved.")


# malaria= 80.83
# tb=85.33
# dengue=89
# hepaB-acc=50%
# hepaB-conf=59%
# asthma=59.5
# GERD=55.16
# IBS=79%
# Anemia=67
# Depression=68.83
# Anxiety=90.33
# Viral fever=84.33
# migraine_acc=50%
# migraine_conf=62%
# Acute Diarrheal Illness=98.66
# COVID=62.16(conf)
# COVID=83.33(acc)

