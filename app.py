import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests

# ---------------- Load data ----------------
book_pivot = pd.read_csv("book_sugg.csv", index_col=0)

# ---------------- Load trained model ----------------
with open("knn_model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------- Helper: Book cover ----------------
def get_book_cover(title):
    url = f"https://openlibrary.org/search.json?title={title}"
    try:
        r = requests.get(url, timeout=5)
        data = r.json()
        if data["docs"]:
            cover_id = data["docs"][0].get("cover_i")
            if cover_id:
                return f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg"
    except:
        pass
    return None

# ---------------- UI ----------------
st.set_page_config(page_title="📚 Book Recommender", layout="wide")
st.title("📚 Book Recommendation System")

search_query = st.text_input("🔍 Search a book")

if search_query:
    books = [b for b in book_pivot.index if search_query.lower() in b.lower()]
else:
    books = book_pivot.index.tolist()

book_name = st.selectbox("Select a book:", books)

if st.button("Recommend"):
    idx = np.where(book_pivot.index == book_name)[0][0]

    distances, indices = model.kneighbors(
        book_pivot.iloc[idx, :].values.reshape(1, -1),
        n_neighbors=min(11, book_pivot.shape[0])
    )

    st.subheader("⭐ Top Similar Books")

    cols = st.columns(5)
    for i in range(1, 6):
        title = book_pivot.index[indices[0][i]]
        similarity = 1 - distances[0][i]
        cover = get_book_cover(title)

        with cols[i - 1]:
            if cover:
                st.image(cover, use_container_width=True)
            st.markdown(f"**{title}**")
            st.caption(f"⭐ Similarity: {similarity:.2f}")

    st.subheader("❤️ You may also like")
    for i in range(6, min(11, len(indices[0]))):
        title = book_pivot.index[indices[0][i]]
        similarity = 1 - distances[0][i]
        st.write(f"❤️ **{title}** (Similarity: {similarity:.2f})")
