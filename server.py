import requests as req
import os
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

load_dotenv()
app = Flask(__name__)
CORS(app)


def fetch_image_from_url(image_url):
    try:
        response = req.get(image_url, timeout=10)
        if response.status_code == 200:
            image_data = base64.b64encode(response.content).decode('utf-8')
            return jsonify({'success': True, 'img_data': image_data})
        else:
            return jsonify({'error': 'Failed to fetch the image. Please check the URL'}), 400
    except req.RequestException as e:
        return {'error': 'Failed to fetch the image. Request exception: ' + str(e)}, 400
    except Exception as e:
        return {'error': 'Failed to fetch the image. Unexpected error: ' + str(e)}, 500


@app.route('/api/fetch_image', methods=["GET", 'POST'])
def fetch_image():
    try:
        image_url = request.get_json().get('image_url')
        print(image_url)
        if not image_url:
            return jsonify({'error': 'Please provide an image URL'}), 400

        result = fetch_image_from_url(image_url)
        return result

    except Exception as e:
        return jsonify({'An error occurred while fetching the image: ': str(e)}), 500


@app.route('/api/transcribe-audio', methods=['POST'])
def upload_audio():
    try:
        audio_file = request.files['audioFile']
        # print(audio_file)
        if not audio_file:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file.save('uploaded_audio.wav')   

        try:
            with open('uploaded_audio.wav', 'rb') as file:
                transcribed_text = openai.Audio.translate(
                    "whisper-1", file)["text"]
                # print(transcribed_text)
        except Exception as e:
            print("Error while parsing the audio file!", e)

        return jsonify({'success': True, "transcript": transcribed_text}), 200

    except Exception as e:
        return jsonify({'error': 'An error occurred while processing the audio: ' + str(e)}), 500


@app.route('/api/process-transcription', methods=['POST'])
def process_transcription():
    try:
        text_message = request.get_json().get('text')
        message = prompt_template.format_messages(
            query=text_message,
            format_instructions=format_instructions
        )
        res = llm(message)
        dictionary = output_parser.parse(res.content)
        return jsonify({'success': True, "llm_output": dictionary}), 200
    except openai.OpenAIError as e:
        return jsonify({'error': 'OpenAI Error: ' + str(e)}), 500
    except Exception as e:
        return jsonify({'error': 'An error occurred while processing the audio: ' + str(e)}), 500


@app.route('/api/health-check', methods=['GET'])
def health_check():
    return jsonify({'status': "ok"}), 200


if __name__ == '__main__':
    llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_KEY"), temperature=0.0)
    openai.api_key = os.getenv("OPENAI_KEY")
    text_template = """\
    For the following search query text that a user wants to search on a real estate \
    property listing website delimited by three backticks extract the following information:

    text: ```{query}```

    {format_instructions}

    Important: In answering questions like city, region, suburb, state you do not need to answer \
    the word itself but for street you have to if it is given in the search query.

    """

    property_action_schema = ResponseSchema(
    name="PropertyAction", 
    description="""You need to extract the information whether the query is for buying, selling or renting a property. \
    If you find that the query text is for buying the property answer "BUY" and if it is for selling \
    answer "SELL" and if it is related to renting then answer "RENT". If you cannot find any relevant information simply return "BUY"."""
    )

    property_type_schema = ResponseSchema(
        name="propertyType", 
        description="""From the query text extract the information about the type of the property mentioned in \
        the text from the list: [House, Unit, Apartment, Studio, Townhouse, Land, Villa, Rural]. If no information is available answer "All"."""
    )

    city_schema = ResponseSchema(
        name="city", 
        description="""From the query text extract the city name from the list [Sydney, Melbourne, Hobart, Darwin, Adelaide, Perth, Brisbane, Canberra] \
        If no information is available answer null."""
    )

    house_num_schema = ResponseSchema(
        name="houseNumber", 
        description="""From the query text extract the apartment or house number from information that the user is searching for and answer it. If no information is \
        available answer null."""
    )

    street_schema = ResponseSchema(
        name="street", 
        description="""From the query text extract the street name from information that the user is searching for and answer it. If no information is \
        available answer null."""
    )

    suburb_schema = ResponseSchema(
        name="suburb", 
        description="""From the query text extract the suburb (Australian) name from information that the user is searching for and answer it. If no information is \
        available answer null."""
    )

    state_schema = ResponseSchema(
        name="state", 
        description="""From the query text extract any one state name from the list: [Australian Capital Territory, New South Wales, Northern Territory, \
        Queensland, South Australia, Tasmania, Victoria, Western Australia]. \If no information is available answer null."""
    )

    postcode_schema = ResponseSchema(
        name="postcode", 
        description="""From the query text extract the postcode (Australian) from information that the user is searching for and answer it. If no information is \
        available answer null."""
    )

    bedrooms_schema = ResponseSchema(
        name="Bedrooms", 
        description="""Extract the number of bedrooms from the text and answer it in integer. If no information is \
        available answer null."""
    )

    bathrooms_schema = ResponseSchema(
        name="Bathrooms", 
        description="""Extract the number of bathrooms from the text and answer it in integer. If no information is \
        available answer null."""
    )

    car_spaces_schema = ResponseSchema(
        name="CarSpaces", 
        description="""Extract the number of car spaces or parkings from the text and answer it in integer. If no information is \
        available answer null."""
    )

    price_range_schema = ResponseSchema(
        name="PriceRange", 
        description="""Extract the price range from the query text that the user is looking for and answer it in the form \
        of an dictionary that includes the Min and Max of the price range in integer. If you don't find any relevant information \
        simply answer null."""
    )

    features_schema = ResponseSchema(
        name="Features", 
        description="""Extract the features from the query text that the user is looking for such as swimming pool, gym etc \
        and answer it as a python list."""
    )

    response_schema = [property_action_schema, property_type_schema, city_schema, house_num_schema, street_schema, suburb_schema, state_schema, postcode_schema, bedrooms_schema, bathrooms_schema, car_spaces_schema, price_range_schema, features_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schema)
    format_instructions = output_parser.get_format_instructions()

    prompt_template = ChatPromptTemplate.from_template(text_template)
    print("loaded")
    app.run(host='0.0.0.0', port=5000, debug=True)
