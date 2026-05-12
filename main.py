# lets first extract info form the pdf
import pymupdf
doc=pymupdf.open("a.pdf")
with open("out.txt","wb") as f:
    for page in doc:
        text=page.get_text().encode("utf-8")
        f.write(text)
        f.write(bytes((12,)))
