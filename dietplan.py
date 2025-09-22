from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel
from typing import List, Optional
import re
from fastapi import FastAPI
import json
import os
import uvicorn

app = FastAPI()

# ----------------- Pydantic Models -----------------
class MealPlan(BaseModel):
    Veg: List[str] = []
    Non_Veg: List[str] = []

class DietPlan(BaseModel):
    Breakfast: MealPlan
    Lunch: MealPlan
    Snacks: List[str] = []
    Dinner: MealPlan
    Foods_to_avoid: List[str] = []
    error: Optional[str] = None

# ----------------- LLM and Parser -----------------
parser = JsonOutputParser(pydantic_object=DietPlan)

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key="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"
)

prompt_template = PromptTemplate(
    template="""
You are a nutrition assistant. Use ONLY the context provided below to suggest a daily diet plan.

Instructions:
1. Suggest meals for Breakfast, Lunch, Snacks, and Dinner.
2. Include only foods that are marked as Favor in the context.
3. Assign foods logically to each meal.
4. Separate vegetarian and non-vegetarian items.
5. Randomly select foods for each meal, without repeating the same food.
6. From Avoid foods, pick a few representative items to list separately.
7. List Maximum 10 Items for each

IMPORTANT: Output must be valid JSON matching this schema exactly:
{{
  "Breakfast": {{ "Veg": [], "Non_Veg": [] }},
  "Lunch": {{ "Veg": [], "Non_Veg": [] }},
  "Snacks": [],
  "Dinner": {{ "Veg": [], "Non_Veg": [] }},
  "Foods_to_avoid": [],
  "error": null
}}

Do NOT include any text outside of this JSON.

{format_instructions}

If data is insufficient, respond as:
{{ "error": "Insufficient Data" }}

Context:
{context}
""",
    input_variables=["context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt_template | llm | parser

# ----------------- Helper Functions -----------------
def clean_foods(text: str) -> str:
    foods_line = text.split("Foods:")[-1]
    foods = [f.strip() for f in foods_line.split(",")]
    cleaned = [re.sub(r"\*.*", "", f).strip() for f in foods if f.strip()]
    return ", ".join(cleaned)

def build_context(docs):
    context_lines = []
    for doc in docs:
        cleaned_foods = clean_foods(doc.page_content)
        context_lines.append(
            f"Category: {doc.metadata.get('category')}, Type: {doc.metadata.get('type')}, Foods: {cleaned_foods}"
        )
    return "\n\n".join(context_lines)


# ----------------Disease Filter------------------
def filter_meal_plan(meal_plan, patient_diseases, food_disease_map):
    # Build a lookup: food -> diseases to avoid
    avoid_dict = {item['Food']: item.get('Avoid_For', []) or [] for item in food_disease_map}

    filtered_plan = {}
    for meal, foods in meal_plan.items():
        if foods is None:
            # If the meal is None, make it empty
            filtered_plan[meal] = [] if meal in ['Snacks', 'Foods_to_avoid'] else {'Veg': [], 'Non_Veg': []}
            continue

        if isinstance(foods, dict):  # Breakfast, Lunch, Dinner (Veg/Non_Veg)
            filtered_sub = {}
            for sub_meal, sub_foods in foods.items():
                sub_foods = sub_foods or []  # ensure not None
                filtered_sub[sub_meal] = [
                    f for f in sub_foods
                    if not any(d in patient_diseases for d in avoid_dict.get(f, []))
                ]
            filtered_plan[meal] = filtered_sub
        else:  # Snacks or Foods_to_avoid (lists)
            foods = foods or []  # ensure not None
            filtered_plan[meal] = [
                f for f in foods
                if not any(d in patient_diseases for d in avoid_dict.get(f, []))
            ]
    return filtered_plan



    # Filter the plan

# ----------------- API Endpoint -----------------
@app.get("/{Dosha}", response_model=DietPlan)
def get_dosha(Dosha: str,disease: Optional[str] = None):        
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    load_db = FAISS.load_local('fiass_index1', embeddings, allow_dangerous_deserialization=True)
    retriever = load_db.as_retriever(search_kwargs={"k": 90})

    dosha_docs = retriever.get_relevant_documents(Dosha)
    dosha_docs1 = [doc for doc in dosha_docs if doc.metadata.get("dosha") == Dosha]

    if not dosha_docs1:
        return DietPlan(
            Breakfast=MealPlan(),
            Lunch=MealPlan(),
            Snacks=[],
            Dinner=MealPlan(),
            Foods_to_avoid=[],
            error=f"No documents found for dosha: {Dosha}"
        )

    context_text = build_context(dosha_docs1)

    # --- Call LLM ---
    llm_raw_output = llm.invoke(prompt_template.format(context=context_text))
    llm_text = llm_raw_output.content
    print("RAW LLM OUTPUT:", llm_text)

    # --- Try Parsing ---
    try:
        response_json = parser.parse(llm_text)
        
    except Exception as e:
        return DietPlan(
            Breakfast=MealPlan(),
            Lunch=MealPlan(),
            Snacks=[],
            Dinner=MealPlan(),
            Foods_to_avoid=[],
            error=f"Parse error: {str(e)}"
        )

    # --- Validate Error Field ---
    if not response_json or response_json.get("error"):
     return DietPlan(
        Breakfast=MealPlan(),
        Lunch=MealPlan(),
        Snacks=[],
        Dinner=MealPlan(),
        Foods_to_avoid=[],
        error="Invalid response from LLM"
    )



    DoshaPlan= DietPlan(**response_json)
    
    if(disease):
        with open('disease_data.json', 'r') as f:
            food_disease_map = json.load(f)

        diet_dict = DoshaPlan.dict()


        filtered_meal_plan = filter_meal_plan(diet_dict, [disease], food_disease_map)
        filtered_meal_plan['error'] = diet_dict.get('error', None)

        return DietPlan(**filtered_meal_plan)

    return DoshaPlan

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("dietplan:app", host="0.0.0.0", port=port)



