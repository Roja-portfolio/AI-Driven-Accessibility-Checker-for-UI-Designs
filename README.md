# AI-Driven-Accessibility-Checker-for-UI-Designs
A rule-based and ML-powered tool to evaluate accessibility in UI screenshots using WCAG guidelines. Supports contrast analysis, alt text checks, and score visualization via Streamlit.
This project is a hybrid **accessibility evaluation tool** for UI screenshots, combining **WCAG 2.1 rule-based checks** with **supervised machine learning models** to assess digital accessibility.

##  Objective ##
To provide a semi-automated system that evaluates the accessibility of UI designs and screenshots, ensuring inclusivity for all users—including those with disabilities—based on **Web Content Accessibility Guidelines (WCAG)**

##  Features ##

** Rule-based evaluation for: 
  - Contrast ratio
  - Use of color
  - Text resizability
  - Presence of alt-text
    
** ML-based accessibility score prediction using:
  - Linear Regression (custom implementation)
  - Random Forest & Logistic Regression (prototype phase)
    
** Outputs:
  - Accessibility score (0–100)
  - Pass/Fail status
  - Downloadable reports
    
** Streamlit-based user interface for quick and interactive testing

##  Dataset ##

** Built using:
  - Manually labeled UI screenshots
  - Synthetic datasets for pattern training
  - 125 screenshots (60% train, 20% validation, 20% test)
    
** Extracted Features:
  - Contrast ratio
  - Font size, spacing
  - Alt-text availability
  - Readability factors

##  Tech Stack ##

- Python (NumPy, Pandas, Pillow, Scikit-learn)
- Streamlit (UI interface)
- Manual ML model (no sklearn used for custom regression)
- WCAG 2.1 Guidelines

## Learnings ##
Implementing a regression model from scratch

Extracting UI/UX features from image data

Combining rule-based logic and ML for real-world evaluation

Building an end-to-end data science pipeline with a frontend

## Future Work ##
Improve ML model using ensemble or deep learning

Expand dataset with real-world annotated UI designs

Support URL-based analysis with browser automation

Accessibility suggestions and auto-fix recommendations

## Reference ##
WCAG 2.1 Guidelines

## Author ##
Roja Shuruthika Kathiravan

M.S. in Computer Science, George Washington University

Course: CSCI 6364 – Machine Learning

Professor: Armin Mehrabian
