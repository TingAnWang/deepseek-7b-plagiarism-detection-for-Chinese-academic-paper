# **Chinese Academic Plagiarism Detection System**

## **Overview**
This system detects potential **plagiarism in Chinese academic text** by combining **FAISS-based embedding similarity search** with an optional **DeepSeek 7B LLM analysis**. It effectively identifies both **direct copying and paraphrased text** by leveraging **semantic similarity retrieval** and **fine-grained text matching**.

## **Features**
- **Paragraph-Level Matching**: Splits large texts into paragraphs for precise similarity detection.
- **FAISS-Based Retrieval**: Uses **FAISS** to store and search **high-dimensional embeddings** of academic texts.
- **Cosine Similarity Computation**: Recomputes similarity scores between query and retrieved text for accuracy.
- **SequenceMatcher Highlighting**: Highlights exact overlapping text segments between query and retrieved documents.
- **Optional DeepSeek 7B Analysis**: Generates a **context-aware verdict** on plagiarism cases.

## **How It Works**
1. **Text Processing**: Reads `.txt` files and splits them into paragraphs.
2. **Vectorization**: Converts each paragraph into embeddings using `sentence-transformers`.
3. **Indexing**: Stores embeddings in a FAISS vector store for fast retrieval.
4. **Query Handling**:
   - When a new text is submitted, it is split into paragraphs.
   - Each paragraph is compared with the most similar stored paragraphs from FAISS.
5. **Plagiarism Detection**:
   - **Cosine similarity** is computed to determine similarity levels.
   - **SequenceMatcher** highlights exact text overlaps.
   - **(Optional) DeepSeek 7B LLM** provides a final **natural language explanation** of suspected plagiarism.

## **Installation & Setup**
### **Requirements**
- Python 3.8+
- FAISS (`pip install faiss-cpu` or `faiss-gpu`)
- `sentence-transformers`
- `transformers` (for DeepSeek 7B, if using LLM analysis)

### **Setup**
```bash
pip install faiss-cpu sentence-transformers transformers
```

## **Usage**
### **Indexing Documents**
1. Place `.txt` academic documents in the specified folder.
2. Run the script to **build the FAISS vector store**:
   ```python
   vector_db = FAISS.from_documents(all_paragraph_docs, embedding_model)
   ```

### **Checking for Plagiarism**
Submit a new text for plagiarism detection:
```python
example_text = "研究方法采用了深度学习..."
check_plagiarism(example_text, top_k=3, similarity_threshold=0.65)
```
The system will:
- Retrieve the **most similar paragraphs**.
- Compute **cosine similarity scores**.
- Highlight **overlapping text**.
- (Optional) Call **DeepSeek 7B** for a **plagiarism verdict**.

### **Example Output**
```bash
Matched Paragraph: "本研究运用了人工智能技术..."
Similarity Score: 0.78
Matched Segments: ["研究方法", "采用了"]
LLM Verdict: "This text is likely a paraphrased version of the retrieved document."
```

## **Customization**
- **Change `similarity_threshold`**: Adjust sensitivity for detecting plagiarism.
- **Modify paragraph splitting logic**: Adjust `split_into_paragraphs()` if documents have different formatting.
- **Fine-tune DeepSeek 7B**: Improve plagiarism detection by training on labeled cases.

## **Future Enhancements**
- Implement **fine-tuning** of DeepSeek 7B for domain-specific plagiarism detection.
- Add **batch processing** for large-scale academic plagiarism analysis.
- Integrate **web UI** for interactive plagiarism checking.
