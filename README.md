# TraceAI

## AI-Powered Requirement Intelligence & Test Case Generation

TraceAI is an AI-powered software testing platform that transforms software requirements and user stories into structured, evaluated, and traceable test cases.

The platform uses the pre-trained **GPT-4o mini** model through the **OpenAI API** to analyze requirements, generate different testing scenarios, evaluate test quality and coverage, and maintain requirement traceability.

TraceAI also provides **Generation History, PDF/Excel export, and Jira/Atlassian integration**, allowing testers to review and polish generated test cases before creating Jira tickets.

---

## 🌐 Live Application

🚀 **Try TraceAI:**  
https://traceai-o4yck4tmruetdpjkunoyjl.streamlit.app

---

## 📂 GitHub Repository

💻 **Source Code:**  
https://github.com/Chirani-Lavanya/TraceAI

---

# 🎯 Project Objective

The goal of TraceAI is to reduce the repetitive effort involved in converting software requirements into comprehensive test cases while keeping the QA engineer in control.

TraceAI helps testers:

- Understand software requirements
- Generate structured test cases
- Improve test coverage
- Maintain requirement traceability
- Evaluate AI-generated results
- Review previous generation runs
- Export testing results
- Create Jira tickets directly from selected test cases

---

# 🚀 Key Features

## 📝 Requirement & User Story Input

Users can choose between:

- Requirement
- User Story

The original input is preserved as the **source of truth** to help maintain requirement grounding and traceability.

---

## 🤖 AI-Powered Test Case Generation

TraceAI uses GPT-4o mini through the OpenAI API to analyze software requirements and generate multiple testing perspectives.

### Supported Test Types

- Functional Test Cases
- Negative Test Cases
- Boundary Value Analysis (BVA)
- Equivalence Partitioning (EP)
- Edge Cases

This helps testers identify scenarios that may be missed during manual test design.

---

## 📊 AI Evaluation

TraceAI evaluates generated test cases using quality and coverage signals including:

- Test Design Coverage
- Requirement Coverage
- Requirement Traceability
- AI Quality Score
- Requirement Grounding

These evaluations help testers review how well the generated test cases align with the original requirement.

---

## 📚 Generation History

TraceAI allows users to revisit previous generation results.

Users can:

- View previous generation runs
- Reopen previous requirements
- Review generated test cases
- Review evaluation results
- Check requirement grounding
- Review coverage and traceability

---

## 📥 PDF & Excel Export

Generated results can be exported for documentation and sharing.

### Excel Workbook

Includes relevant:

- Requirement information
- Evaluation results
- Test cases
- Traceability information

### PDF Report

Provides a formatted report containing the generated testing information and evaluation results.

---

## 🔗 Jira / Atlassian Integration

TraceAI connects generated test cases with the Jira workflow.

Users can:

1. Select a generated test case
2. Review the Jira ticket
3. Polish and customize the ticket content
4. Create the Jira issue directly

This reduces the need to manually copy test-case information into Jira.

📌##Project Summary

TraceAI transforms software requirements and user stories into intelligent, evaluated, and traceable test cases while helping QA engineers reduce repetitive test-design effort.

The platform combines:

AI Test Generation + Evaluation + Requirement Grounding + Traceability + History + PDF/Excel Export + Jira Integration into a unified software testing workflow.

# 🏗️ System Architecture

TraceAI follows a layered architecture connecting the user interface, backend services, AI model, database, and external Jira integration.

<img width="1536" height="1024" alt="System Architecture" src="https://github.com/user-attachments/assets/1cceee2c-4402-4f36-b99c-0c103268b4ba" />

## Architecture Summary
TraceAI connects the frontend, backend, AI model, database, and external Jira integration into one end-to-end testing workflow.

👩‍💻 Author

Chirani Lavanya
Ascentic AI Launch Pad 2026 -- Project Submission
