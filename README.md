# 🏷️ AI-Driven Image-to-IMDB Auto-Fill

> Automatically extract all 10 Item Master Database attributes from product images using OCR + NLP — no manual data entry needed.

## 🎯 Problem

Manually cataloguing FMCG products into an Item Master Database (IMDB) is slow, error-prone, and expensive. A single product takes ~8 minutes to enter manually. With hundreds of products, this becomes a bottleneck for distributors and retailers.

## 💡 Solution

Upload a product image → AI extracts all 10 IMDB attributes in seconds → Review, edit, and export to CSV or Excel.

## 🤖 How It Works

| Step | Technology | What It Does |
|------|-----------|--------------|
| 1 | **pyzbar** | Decodes barcodes directly from the image (EAN-8, EAN-13, UPC) |
| 2 | **OpenCV** | Grayscale conversion + adaptive thresholding for clearer text |
| 3 | **Tesseract OCR** | Extracts all visible text from the product label |
| 4 | **NLP / Regex** | Parses OCR text into structured IMDB fields |
| 5 | **Keyword CNN** | Classifies category type & segment type from extracted text |

## 📋 10 IMDB Attributes Extracted

| Field | Example |
|-------|---------|
| Barcode | 6030057221077 |
| Category Type | Personal Care |
| Segment Type | Bar Soap |
| Manufacturer | Meiji Ghana Limited |
| Brand | MOK |
| Product Name | MOK Fine Soap Rose |
| Weight & Unit | 100g |
| Packaging Type | Cardboard Box |
| Country of Origin | Ghana |
| Promotional Messages | Natural and fresh douceur |

## 🚀 Live Demo

**[Try it here →](https://competitionimage-96uoydqx5txmk7ellah8eo.streamlit.app)**

- Turn on **Demo Mode** to see 4 real Ghanaian products instantly
- Or upload any product image to extract attributes live

## 📊 Performance

- ⏱️ **~12 seconds per product** (vs ~8 minutes manually)
- ⚡ **97% time saving** per product catalogued
- ✅ Confidence scoring on all 10 fields — flags low-confidence extractions for human review

## 🗂️ Project Structure

```
├── app.py              # Streamlit web application
├── testing.ipynb       # Model testing & evaluation notebook (EasyOCR + EfficientNet CNN)
├── product_images/     # Sample Ghanaian FMCG product images
├── requirements.txt    # Python dependencies
└── packages.txt        # System packages (Tesseract, ZBar)
```

## 🛠️ Run Locally

```bash
git clone https://github.com/myna23/Competition_Image.git
cd Competition_Image
pip install -r requirements.txt
streamlit run app.py
```

## 📓 Testing Notebook

The `testing.ipynb` notebook demonstrates the full ML pipeline:
- **EasyOCR (CRAFT + CRNN)** for text detection and recognition
- **EfficientNet-B0 CNN** (ImageNet pre-trained) for category classification
- Confidence heatmaps and field-level performance analysis
- Export to CSV and Excel

## 📦 Dataset

169 real FMCG product images from Ghanaian retail stores across multiple categories:
- Personal Care (soaps, lotions)
- Food & Beverage (juices, seasonings, chocolate drinks)
