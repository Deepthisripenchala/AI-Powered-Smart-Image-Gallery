# =========================================================
# app.py — Final Integrated Gemini AI Gallery & Planner
# =========================================================
import streamlit as st
import os
import sqlite3
import pickle
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import requests
import uuid
from typing import List, Tuple, Optional, Dict, Any
import io
import urllib3
from dataclasses import dataclass

# Suppress warnings for requests/urllib3 in a development environment
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# =====================================================================
# CONFIGURATION & KEY MANAGEMENT
# =====================================================================

# 🔑 Your Gemini API Key
GEMINI_API_KEY = "AIzaSyCzFC4fLcZDS-sp8f9bB0qZN26L9ZE_OyU"  # 👈 Paste Gemini Key here

# --- MODEL DEFINITIONS ---
MODEL_FLASH = "gemini-2.5-flash" 
MODEL_EMBEDDING = "text-embedding-004"
EMBEDDING_DIM = 768
# -------------------------

# Data paths
DATA_DIR = "data"
IMG_DIR = os.path.join(DATA_DIR, "images")
DB_PATH = os.path.join(DATA_DIR, "gallery.db") 

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

CATEGORIES = ["Cars", "Houses", "Locations", "People", "Belongings", "Clothes"]

# Set Streamlit config early
st.set_page_config(page_title="Gemini AI Agent", layout="wide")


# =====================================================================
# DATABASE SETUP
# =====================================================================

SCHEMA_UNIFIED = """
CREATE TABLE IF NOT EXISTS unified_images (
    id TEXT PRIMARY KEY,
    filename TEXT,
    category TEXT,
    tags TEXT,
    caption TEXT,
    emotion TEXT,
    embedding BLOB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

@dataclass
class ImageRecord:
    id: str
    filename: str
    category: str
    tags: str
    caption: str
    emotion: str
    embedding: Optional[np.ndarray]

def get_conn() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)

def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA_UNIFIED)
        conn.commit()

init_db()


# =====================================================================
# GEMINI API FUNCTIONS
# =====================================================================

def gemini_configure():
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE" or not GEMINI_API_KEY:
        st.error("🚨 Please set your Gemini API Key in the `app.py` script.")
        st.stop()
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception as e:
        st.error(f"🚨 Gemini API Configuration Failed. Error: {e}")
        st.stop()

def gemini_caption_image(image_path):
    img = Image.open(image_path)
    model = genai.GenerativeModel(MODEL_FLASH) 
    response = model.generate_content([img, "Describe this image briefly in one sentence."])
    return (response.text or "No caption generated").strip()

def gemini_detect_emotion(text):
    model = genai.GenerativeModel(MODEL_FLASH)
    prompt = f"Return only one word describing the emotion or mood of this sentence: '{text}'"
    response = model.generate_content(prompt)
    return (response.text or "neutral").strip().lower()

def gemini_get_embedding_text(text):
    res = genai.embed_content(
        model=MODEL_EMBEDDING,
        content=text,
        task_type="retrieval_document"
    )
    return np.array(res['embedding']) 

def gemini_get_embedding_image_for_db(image_path):
    """
    Uses Gemini-Flash to describe the image, then uses the embedding model
    to get a semantic vector for storage/search.
    """
    img = Image.open(image_path)
    model = genai.GenerativeModel(MODEL_FLASH)
    
    # Generate a rich description for vectorization
    response = model.generate_content([img, "Describe this image in detail, listing key objects, colors, and context. Use this description solely for semantic similarity search purposes."])
    
    if response.text:
        try:
            vec = gemini_get_embedding_text(response.text)
        except Exception as e:
            st.warning(f"Embedding failed: {e}. Using zero vector.")
            vec = np.zeros(EMBEDDING_DIM) 
    else:
        st.warning("Model failed to describe the image. Using zero vector.")
        vec = np.zeros(EMBEDDING_DIM) 
        
    return vec


# =====================================================================
# DATABASE & SEARCH OPERATIONS
# =====================================================================

def save_unified_image(filename, category, tags, caption, emotion, embedding):
    img_id = str(uuid.uuid4())
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT id FROM unified_images WHERE filename = ?", (filename,))
    if cursor.fetchone():
        st.warning(f"Image '{filename}' already exists. Skipping.")
        conn.close()
        return

    conn.execute(
        "INSERT INTO unified_images (id, filename, category, tags, caption, emotion, embedding) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (img_id, filename, category, tags, caption, emotion, pickle.dumps(embedding))
    )
    conn.commit()
    conn.close()

@st.cache_data(show_spinner="Loading image records and rebuilding vector index...")
def load_all_unified_data(filter_categories: Optional[List[str]] = None):
    conn = sqlite3.connect(DB_PATH)
    
    # --- Category Filtering Logic ---
    if filter_categories and filter_categories != CATEGORIES:
        qmarks = ",".join(["?"] * len(filter_categories))
        query = f"SELECT id, filename, category, tags, caption, emotion, embedding FROM unified_images WHERE category IN ({qmarks}) ORDER BY created_at DESC"
        params = filter_categories
    else:
        query = "SELECT id, filename, category, tags, caption, emotion, embedding FROM unified_images ORDER BY created_at DESC"
        params = []

    rows = conn.execute(query, params).fetchall()
    conn.close()
    
    records = []
    ids, embs = [], []
    
    for r in rows:
        rec = ImageRecord(id=r[0], filename=r[1], category=r[2], tags=r[3], caption=r[4], emotion=r[5], embedding=None)
        
        if r[6]:
            try:
                emb = pickle.loads(r[6])
                if emb.shape[0] == EMBEDDING_DIM:
                    rec.embedding = emb
                    ids.append(rec.id)
                    embs.append(emb)
            except Exception:
                pass
        records.append(rec)
    
    if not embs:
         return records, np.empty((0, EMBEDDING_DIM)), []
         
    return records, np.array(embs), ids


def cosine_search(query_vec, db_vecs, db_ids, top_k=6):
    """Performs cosine similarity search using numpy/sklearn."""
    if db_vecs.shape[0] == 0:
        return []
        
    sims = cosine_similarity([query_vec], db_vecs)[0]
    top_k = min(top_k, len(sims)) 
    idxs = np.argsort(sims)[::-1][:top_k]
    
    return [(db_ids[i], sims[i]) for i in idxs]

def retrieve_by_query(query: str, records: List[ImageRecord], db_vecs: np.ndarray, db_ids: List[str], top_k: int = 5) -> List[Tuple[ImageRecord, float]]:
    """Retrieves records by text query against the vector index."""
    if db_vecs.shape[0] == 0:
        return []
        
    q_emb = gemini_get_embedding_text(query)
    
    hits = cosine_search(q_emb, db_vecs, db_ids, top_k=top_k)
    if not hits:
        return []

    rec_map = {r.id: r for r in records}
    
    results = []
    for uid, score in hits:
        if uid in rec_map:
            results.append((rec_map[uid], score))
            
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def retrieve_by_image(image_path: str, records: List[ImageRecord], db_vecs: np.ndarray, db_ids: List[str], top_k: int = 5) -> List[Tuple[ImageRecord, float]]:
    """Retrieves records by image query against the vector index."""
    if db_vecs.shape[0] == 0:
        return []
        
    # Get the multimodal embedding for the query image
    q_emb = gemini_get_embedding_image_for_db(image_path)
    
    hits = cosine_search(q_emb, db_vecs, db_ids, top_k=top_k)
    if not hits:
        return []

    rec_map = {r.id: r for r in records}
    
    results = []
    for uid, score in hits:
        if uid in rec_map:
            results.append((rec_map[uid], score))
            
    results.sort(key=lambda x: x[1], reverse=True)
    return results


# =====================================================================
# UNSPLASH API (Unchanged)
# =====================================================================

def search_images_unsplash(query: str, num_results: int = 5, access_key: Optional[str] = None) -> List[str]:
    if not access_key:
        st.warning("❗ Unsplash Access Key not set in the sidebar.")
        return []
    try:
        url = "https://api.unsplash.com/search/photos"
        params = {"query": query, "per_page": num_results, "client_id": access_key}
        resp = requests.get(url, params=params, timeout=10, verify=False) 
        if resp.status_code == 401:
             st.error("Unsplash API Error: Invalid Access Key (401). Check the key in the sidebar.")
             return []
        if resp.status_code != 200:
            st.error(f"Unsplash API error: {resp.status_code}. Check API rate limits or query.")
            return []
        data = resp.json()
        return [r["urls"]["regular"] for r in data.get("results", []) if "urls" in r]
    except Exception as e:
        st.error(f"Error fetching images from Unsplash: {e}")
        return []

def display_unsplash_images(query: str, unsplash_key: str, num_results: int = 6):
    st.info(f"🔍 Searching Unsplash for '{query}'...")
    urls = search_images_unsplash(query, num_results=num_results, access_key=unsplash_key)
    if not urls:
        st.warning("No results found on Unsplash.")
        return
    cols = st.columns(3)
    for i, url in enumerate(urls):
        with cols[i % 3]:
            st.image(url, caption=f"Unsplash result {i+1}", use_container_width=True)


# =====================================================================
# OUTFIT PLANNER LOGIC (Unchanged)
# =====================================================================

def outfit_and_packing_suggestions(query: str) -> Dict[str, Any]:
    q = query.lower()
    cool = any(k in q for k in ["shimla", "mountain", "snow", "cold", "winter", "freezing"])
    hike = any(k in q for k in ["hike", "trek", "trail", "walk"])
    beach = any(k in q for k in ["beach", "swim", "coast", "tropical", "hot"])
    
    layers = []
    if cool:
        layers = ["Thermal wear", "Fleece jacket", "Waterproof shell"]
    elif beach:
        layers = ["Swimwear", "Light linen shirt"]
    else:
        layers = ["Casual shirt/top", "Light jacket/windbreaker"]

    lower = ["Trek pants" if hike else "Chinos/Jeans"]
    shoes = ["Hiking boots" if hike else "Sandals/Sneakers" if beach else "Casual shoes"]
    add = []
    if cool:
        add = ["Beanie", "Gloves", "Scarf"]
    elif beach:
        add = ["Sunscreen", "Hat", "Sunglasses"]
    else:
        add = ["Cap", "Sunglasses"]
        
    return {
        "outfit": layers + lower + shoes, 
        "accessories": add,
        "backpack": ["Water bottle", "Snacks", "Portable charger"]
    }


# =====================================================================
# STREAMLIT UI - Pages
# =====================================================================

# --- Page: Gallery (Gemini) ---
def page_gemini_gallery(unsplash_key):
    gemini_configure()
    
    st.title("1. 🖼️ AI Gallery & Chat (Gemini)")
    st.markdown("---")
    
    # ADDED VISUAL SEARCH TAB (Index 2)
    tabs = st.tabs(["📥 Analyze & Upload", "🔍 Text Search", "🧩 Visual Search", "💬 Chat", "📊 Insights"])
    
    # Load ALL records by default for Gallery Chat/Insights/Upload processing
    records_all, embs_all, ids_all = load_all_unified_data(filter_categories=None) 

    # --- Upload Tab (Index 0) ---
    with tabs[0]:
        st.header("Upload, Analyze, and Index")
        file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="gemini_upload")
        if file:
            image_path = os.path.join(IMG_DIR, file.name)
            Image.open(file).save(image_path)
            
            st.image(image_path, caption="Uploaded Image", use_column_width=True)

            col1, col2 = st.columns(2)
            with col1:
                category = st.selectbox("Category", CATEGORIES, key="new_cat")
            with col2:
                tags = st.text_input("Tags (e.g., vintage, sunset, beach)", key="new_tags")
            
            if st.button("Analyze & Save"):
                with st.spinner("Analyzing image with Gemini..."):
                    caption = gemini_caption_image(image_path)
                with st.spinner("Detecting emotion and mood..."):
                    emotion = gemini_detect_emotion(caption)
                with st.spinner("Creating multimodal embedding for search..."):
                    embedding = gemini_get_embedding_image_for_db(image_path) 

                save_unified_image(file.name, category, tags, caption, emotion, embedding)
                st.success("✅ Image fully processed and saved!")
                st.write(f"**Generated Caption:** {caption}")
                st.write(f"**Detected Emotion:** **{emotion}**")
                st.write("---")
                st.info("Reloading application to update search index...")
                st.cache_data.clear()
                st.rerun()
                

    # --- Text Search Tab (Index 1) ---
    with tabs[1]:
        st.header("🔍 Semantic Search")
        
        selected_category = st.selectbox(
            "Filter Search by Category:",
            ["ALL"] + CATEGORIES,
            key="search_filter_cat"
        )
        
        query = st.text_input("Search for an image by description:", key="gemini_search_query")
        topk = st.slider("Number of results", 1, 10, 5, key="search_topk")
        source = st.radio("Search Source", ["My Gallery", "Unsplash"], key="general_source")
        
        if st.button("Execute Text Search"):
            if source == "Unsplash":
                display_unsplash_images(query, unsplash_key, num_results=topk)
            else:
                if selected_category == "ALL":
                    filter_cats = None
                    display_records, display_embs, display_ids = records_all, embs_all, ids_all
                else:
                    filter_cats = [selected_category]
                    # We reload data here only if a specific filter is selected
                    display_records, display_embs, display_ids = load_all_unified_data(filter_categories=filter_cats)
                
                if display_embs.shape[0] == 0:
                    st.warning(f"No indexed images found for the category '{selected_category}'.")
                else:
                    with st.spinner(f"Searching {len(display_records)} items in your gallery..."):
                        results = retrieve_by_query(query, display_records, display_embs, display_ids, top_k=topk)
                        
                    st.subheader(f"Results from {selected_category}:")
                    cols = st.columns(3)
                    for i, (r, score) in enumerate(results):
                        with cols[i % 3]:
                            st.image(os.path.join(IMG_DIR, r.filename),
                                      caption=f"Score: {score:.3f} | {r.caption} ({r.emotion})")

    # --- Visual Search Tab (Index 2) ---
    with tabs[2]:
        st.header("🧩 Visual Similarity Search")
        
        ref_file = st.file_uploader("Upload image to search your database with:", type=["jpg", "jpeg", "png"], key="visual_upload")
        topk = st.slider("Number of visual results", 1, 10, 5, key="visual_topk")
        
        if st.button("Find Similar Images"):
            if not ref_file:
                st.warning("Please upload a reference image first.")
            elif embs_all.shape[0] == 0:
                st.warning("Your gallery is empty. Please upload some images first.")
            else:
                # Save the uploaded file temporarily to analyze
                ref_path = os.path.join(IMG_DIR, "query_ref_" + ref_file.name)
                Image.open(ref_file).save(ref_path)

                with st.spinner("Computing reference embedding and searching..."):
                    # Retrieve the top similar images using the image embedding function
                    results = retrieve_by_image(ref_path, records_all, embs_all, ids_all, top_k=topk)
                
                if results:
                    st.subheader("Results:")
                    cols = st.columns(3)
                    for i, (r, score) in enumerate(results):
                        with cols[i % 3]:
                            st.image(os.path.join(IMG_DIR, r.filename),
                                      caption=f"Score: {score:.3f} | {r.caption}", use_container_width=True)
                else:
                    st.info("No similar images found in your gallery.")
                
                # Clean up the temporary query file
                os.remove(ref_path)

    # --- Chat Tab (Index 3, unchanged) ---
    with tabs[3]:
        st.header("💬 Chat with Your Gallery")
        q = st.text_input("Ask about your gallery (e.g., 'Do I have more happy or sad photos?'):", key="gemini_chat_query")
        if st.button("Ask Gemini Chat"):
            if not records_all:
                st.warning("No images yet.")
            else:
                context = "\n".join([f"Image {i+1} (Tags: {r.tags}): {r.caption} (Emotion: {r.emotion})" for i, r in enumerate(records_all)])
                
                model = genai.GenerativeModel(MODEL_FLASH)
                prompt = (
                    f"You are an image gallery assistant. Answer the user's question only based on the provided gallery context.\n"
                    f"User question: {q}\n\n"
                    f"--- Gallery Inventory (Total {len(records_all)} images) ---\n{context}\n-----------------------------------\n\n"
                    f"Answer clearly and briefly, referencing the themes, tags, or emotions."
                )
                
                with st.spinner("Thinking..."):
                    response = model.generate_content(prompt)
                    st.write(response.text)

    # --- Insights Tab (Index 4, unchanged) ---
    with tabs[4]:
        st.header("📊 Gallery Insights")
        st.metric("Total Indexed Images", len(records_all))
        
        if records_all:
            emo_counts = {}
            for r in records_all:
                emo_counts[r.emotion] = emo_counts.get(r.emotion, 0) + 1
                
            st.subheader("Emotion Distribution")
            st.bar_chart(emo_counts)
            
            st.subheader("Recent Uploads")
            cols = st.columns(4)
            for i, r in enumerate(records_all[:8]):
                with cols[i % 4]:
                    st.image(os.path.join(IMG_DIR, r.filename), width=180)
                    st.caption(f"{r.caption} (Emotion: **{r.emotion}**)")
        else:
            st.info("Upload some images to the 'Analyze & Upload' tab to see insights!")


# --- Page: Outfit Planner (Unchanged Logic, Correctly Filtered) ---
def page_outfit_planner(unsplash_key: str):
    st.title("2. 🎒 Outfit Planner & Visualizer")
    st.markdown("---")
    
    OUTFIT_CATEGORIES = ["Clothes", "Belongings"]
    records, embs, ids = load_all_unified_data(filter_categories=OUTFIT_CATEGORIES)

    st.info(f"Local gallery search is filtered to categories: **{', '.join(OUTFIT_CATEGORIES)}**")

    q = st.text_input("Tell me about your trip/event:", "I am going hiking in the mountains and need cold-weather clothes.", key="planner_q")
    source = st.radio("Image Source for Visualization:", ["My Gallery", "Unsplash"], key="planner_source")
    
    if st.button("Suggest Outfit & Find Matches"):
        plan = outfit_and_packing_suggestions(q)
        
        st.subheader("✅ Suggested Outfit")
        st.markdown("---")
        for item in plan["outfit"]:
            st.markdown(f"**•** {item}")
        
        st.subheader("🧳 Suggested Accessories/Packing")
        st.markdown("---")
        for item in plan["accessories"]:
            st.markdown(f"**•** {item}")
        for item in plan["backpack"]:
            st.markdown(f"**•** {item}")

        st.subheader("🖼️ Visual Suggestions")
        st.markdown("---")

        if source == "Unsplash":
            with st.spinner("Searching Unsplash for visual examples..."):
                display_unsplash_images(q + " outfit", unsplash_key, num_results=6)
        else:
            if embs.shape[0] == 0:
                 st.info(f"Your filtered gallery is empty. Please upload images categorized as {OUTFIT_CATEGORIES} first.")
            else:
                with st.spinner(f"Searching {len(records)} matching items in your gallery..."):
                    search_query = f"{q} clothing item, accessories, shoes, bag"
                    results = retrieve_by_query(search_query, records, embs, ids, top_k=6)
                    
                if results:
                    st.subheader("Matching items from **Your Gallery**:")
                    cols = st.columns(3)
                    for i, (r, score) in enumerate(results):
                        img_filename = os.path.join(IMG_DIR, r.filename)
                        
                        with cols[i % 3]:
                            st.image(img_filename,
                                      caption=f"{r.tags} | Score: {score:.3f}", use_container_width=True)
                else:
                    st.info("No close matches found in your filtered gallery. Try switching to Unsplash for general ideas.")


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================

def main():
    
    with st.sidebar:
        st.title("🛠️ Tool Configuration")
        st.markdown("---")

        if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE" or not GEMINI_API_KEY:
             st.error("Set GEMINI_API_KEY in script.")
             
        unsplash_key = st.text_input("Unsplash API Key", type="password", help="Required for Unsplash searches.")
        st.markdown("---")
        
        st.subheader("Navigation")
        page = st.radio("Select Feature", ["1. 🖼️ Gallery (Gemini)", "2. 🎒 Outfit Planner"], index=0)

    # --- Router ---
    if page == "1. 🖼️ Gallery (Gemini)":
        page_gemini_gallery(unsplash_key)
    elif page == "2. 🎒 Outfit Planner":
        page_outfit_planner(unsplash_key)

if __name__ == "__main__":
    main()