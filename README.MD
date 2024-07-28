# Job-Matching RAG Application

This Streamlit application helps match job candidates with job descriptions using Retrieval-Augmented Generation (RAG) with Google's Generative AI.

## Features

- Input job descriptions and retrieve suitable candidates from a dataset.
- Uses Google's Generative AI for embedding and chat functionality.
- Simple and user-friendly interface.

## Setup

### Prerequisites

- Python 3.8 or higher
- Streamlit
- Google API Key (for Google's Generative AI)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/job-matching-rag-application.git
    cd job-matching-rag-application
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. Set up your Google API key in the environment variable `GOOGLE_API_KEY` or input it in the sidebar of the application.

2. Run the Streamlit application:

    ```bash
    streamlit run app.py
    ```

3. Follow the instructions on the page to input your job description and get the matched candidates.

### File Structure

- `app.py`: The main Streamlit application file.
- `requirements.txt`: Lists the Python packages required to run the application.
- `README.md`: Provides an overview and setup instructions for the application.

## How to Use

1. Enter your Google API Key in the sidebar.
2. Upload a CSV file containing candidate details.
3. Enter the job description in the text input field.
4. Click the 'Get Results' button to retrieve suitable candidates.

## Example

Job Description: *We are looking for a skilled UI Developer to join our dynamic team. The ideal candidate will have a strong background in front-end development, with proficiency in HTML, CSS, JavaScript, and modern frameworks like React or Angular. Your primary responsibility will be to create visually appealing and user-friendly web interfaces that enhance user experience and align with our brand guidelines.*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
