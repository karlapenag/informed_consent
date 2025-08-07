import fitz  # PyMuPDF
import re

# replacement patterns are specific to Euclid trial. This can be modified

replacements = {
    r'\(insert your\s*PI’s Name here\)|\[INSERT PI NAME\]|\(insert PI’s name here\)|\[insert your\s*PI’s name here\]|\(PI’s Name\)|\[INSERT PRINCIPAL INVESTIGATOR\s*NAME\]': 'John A. Smith',
    r'\(insert\s*PI’s number here with area code\)|\(PI’s Number with Area Code\)': '(919) 555-1234',
    r'\(insert PI’s 24-hour number here\s*with area code\)|\(PI’s 24-hour Number with Area Code\)': '(919) 555-5678',
    r'\[insert PI’s\s*address here\]': '123 Research Lane, Durham, NC 27701',
    r'\[INSERT\s*INSTITUTION/FACILITY\s*NAME\]|\[INSERT\s*INSTITUTION/HEALTH CARE PROVIDER NAME\]|\[INSERT INSTITUTION NAME\]': 'Duke University Medical Center',
    r'\[INSERT INSTITUTION ADDRESS\]': '456 Clinical Research Blvd, Durham, NC 27710',
    r'\[INSERT IRB NAME\]|\(insert\s*your IRB name here\)|\[INSERT INSTITUTION IRB NAME\]': 'Duke Health Institutional Review Board',
    r'\(insert your IRB’s phone number\s*with area code\)': '(919) 555-7890',
    r'\[INSERT SITE\s*SPECIFICS\]|\(insert your site info here\)': 'Duke Clinical Research Institute (DCRI)',
    r'\[INSERT OTHERS AS APPROPRIATE\]': 'National Institutes of Health (NIH)',
    r'\[INSERT INFORMATION\s*WHERE CODE WILL BE KEPT\]': 'Duke Health Secure Data Repository',
    r'\[his/her\]|\[him/her\]|\[His/her\]': 'his',
    r'Dr\. ________': 'Dr. John A. Smith'
}

def extract_text_without_header_footer_and_replace(pdf_path, replacement=True,
                                                   header_height_in_inches=1.7,
                                                   footer_height_in_inches=1,
                                                   last_page_footer_height_in_inches=2):
    header_height = header_height_in_inches * 72  # Convert inches to points
    footer_height = footer_height_in_inches * 72
    last_page_footer_height = last_page_footer_height_in_inches * 72
    text = ""

    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc):
            page_rect = page.rect
            current_footer_height = last_page_footer_height if page_num == len(doc) - 1 else footer_height

            if page_num == 0:
                no_header_footer_rect = fitz.Rect(page_rect.x0, page_rect.y0, page_rect.x1, page_rect.y1 - current_footer_height)
            else:
                no_header_footer_rect = fitz.Rect(page_rect.x0, page_rect.y0 + header_height, page_rect.x1, page_rect.y1 - current_footer_height)

            page_text = page.get_text("text", clip=no_header_footer_rect)

            if replacement:
                for pattern, repl in replacements.items():
                    page_text = re.sub(pattern, repl, page_text)

            text += page_text + "\n"

    return text