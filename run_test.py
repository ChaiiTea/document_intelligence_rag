from src.pipeline import DocumentIntelligencePipeline

pipeline = DocumentIntelligencePipeline.from_config("configs/config.yaml")
doc = pipeline.run("G:\\pilot\\project_2\\data\\raw\\faq-composition-levy-revised.pdf")
print(doc.to_dict())