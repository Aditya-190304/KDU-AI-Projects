python -m medical_extraction.cli.extract_document \
  --input ./sample_docs/report.pdf \
  --output ./output/report_extraction.json \
  --debug-dir ./output/debug/report \
  --save-debug-images true \
  --enable-medical-ner true \
  --device cpu
