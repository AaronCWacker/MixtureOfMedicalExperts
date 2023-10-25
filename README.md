# MixtureOfMedicalExperts
MoE Mode with Mixture of Experts to Support Health Care Scenarios in Multi Agent Systems

# Prompt

Create a mixture of experts model based on below CHARMSED model where you have eight roles that each contain a multiple 10 line question form which presents a system prompt for a role.  Set the roles so that they follow a Doctor specialty expert role model where there are eight general experts in doctor specialties and nursing specialties that generate output for the following tasks:  1. Triage and Check in for patients, 2. a lab analyst that studies condition lists and produces a lab, test, summary.  3. a medicine specialist that understands conditions over time against controlled and uncontrolled status with drugs at the time.  4. A service expert that is empathetic to patient cost and invasiveness of treatments, knows treatment and can list a top ten service codes for a list of diagnosis conditions.  5. A level of care expert that understands the mixture of doctor specialties a given patient needs and can name the specialty taxonomy code and descriptions.  6. A terminology expert which mastery over USMLE for medical licensing which is also a psychiatrist and neurologist that can help augment intelligence of the doctor working with this multi system agent.  7. A chief medical officer. and 8. a medical director team with oversight of the payer, the patient, and the collection of providers and facilities like medical centers within driving distance of any patient by state.  Here is the CHARMSED model:  Harness the power of AI with the CHARMSED framework.

This suite of roles brings together a comprehensive set of AI capabilities, tailored for diverse tasks:
Coder 💻: Craft pythonic solutions with precision.
Humanities Expert 📚: Dive deep into arts, literature, and history.
Analyst 🤔: Derive insights through logical reasoning.
Roleplay Expert 🎭: Mimic behaviors or adopt personas for engaging interactions.
Mathematician ➗: Crunch numbers and solve mathematical enigmas.
STEM Expert 🔬: Navigate through the realms of Science, Technology, Engineering, and Mathematics.
Extraction Expert 🔍: Extract concise information with a laser-focus.
Drafter 📝: Generate textual content and narratives with flair.
Empower your tasks with the perfect AI role and unleash the magic of CHARMSED!

Select AI Role:

# Program:

import streamlit as st

def triage_checkin():
    st.write("### Triage and Check-in Expert 🚑")
    for i in range(1, 11):
        st.text_input(f"Question {i} for Triage")

def lab_analyst():
    st.write("### Lab Analyst 🧪")
    for i in range(1, 11):
        st.text_input(f"Question {i} for Lab Analysis")

def medicine_specialist():
    st.write("### Medicine Specialist 💊")
    for i in range(1, 11):
        st.text_input(f"Question {i} for Medicine")

def service_expert():
    st.write("### Service Expert 💲")
    for i in range(1, 11):
        st.text_input(f"Question {i} for Service")

def care_expert():
    st.write("### Level of Care Expert 🏥")
    for i in range(1, 11):
        st.text_input(f"Question {i} for Level of Care")

def terminology_expert():
    st.write("### Terminology Expert 📚")
    for i in range(1, 11):
        st.text_input(f"Question {i} for Terminology")

def cmo():
    st.write("### Chief Medical Officer 🩺")
    for i in range(1, 11):
        st.text_input(f"Question {i} for CMO")

def medical_director():
    st.write("### Medical Director Team 🏢")
    for i in range(1, 11):
        st.text_input(f"Question {i} for Medical Director")

def main():
    st.title("Mixture of Medical Experts Model")
    st.write("Harness the power of AI with this specialized healthcare framework! 🎉")

    role = st.selectbox("Select AI Role:", [
        "Triage and Check-in Expert",
        "Lab Analyst",
        "Medicine Specialist",
        "Service Expert",
        "Level of Care Expert",
        "Terminology Expert",
        "Chief Medical Officer",
        "Medical Director Team"
    ])

    if role == "Triage and Check-in Expert":
        triage_checkin()
    elif role == "Lab Analyst":
        lab_analyst()
    elif role == "Medicine Specialist":
        medicine_specialist()
    elif role == "Service Expert":
        service_expert()
    elif role == "Level of Care Expert":
        care_expert()
    elif role == "Terminology Expert":
        terminology_expert()
    elif role == "Chief Medical Officer":
        cmo()
    elif role == "Medical Director Team":
        medical_director()

if __name__ == "__main__":
    main()

