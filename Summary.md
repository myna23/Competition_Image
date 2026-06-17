AI-Driven Image-to-IMDB Auto-Fill

An AI-powered tool that automatically extracts all 10 Item Master Database (IMDB) attributes from a single product image — eliminating manual data entry for FMCG cataloguing.

Model Architecture

The system is built on a two-model pipeline evaluated in a Jupyter notebook and deployed as a live Streamlit application:

EasyOCR (CRAFT + CRNN) — Text detection and recognition. CRAFT locates text regions in the image; CRNN reads each region and returns text with confidence scores. Used to extract brand, weight, manufacturer, barcode, country of origin, and promotional messages from product labels.

EfficientNet-B0 CNN (ImageNet pre-trained, 5.3M parameters) — Image classification for category and segment type. Top-k ImageNet predictions are mapped to an FMCG category taxonomy (Personal Care, Food & Beverage, Healthcare, etc.) and segment (Bar Soap, Seasonings & Spices, Cough & Cold, etc.).

Production Pipeline

The deployed app extends the core models with additional components: pyzbar for direct barcode decoding (EAN-8/13, UPC) with Open Food Facts enrichment; OCR.space cloud OCR (Engine 2) for higher accuracy on complex coloured labels; OpenCV for adaptive thresholding pre-processing; and a custom NLP/regex layer for multi-strategy brand detection, company-name recognition, packaging classification (27 types), and country normalisation.

Performance

~12 seconds per product vs. ~8 minutes manually — 97% time saving
Confidence scoring on all 10 fields with visual flags for human review
Bulk upload with CSV and Excel export
