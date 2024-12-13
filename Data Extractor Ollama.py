#ollama run nuextract

Prompt:
### Template:
{
    "Model": {
        "Name": "",
        "Number of parameters": "",
    },
    "Usage": {
        "Use case": [],
        "Licence": ""
    }
}
### Example:
{
    "Model": {
        "Name": "Llama3",
        "Number of parameters": "8 billion",
    },
    "Usage": {
        "Use case":[
			"chat",
			"code completion"
		],
        "Licence": "Meta Llama3"
    }
}



from ollama import chat
from pydantic import BaseModel

class Equipamento(BaseModel):
    Name: str                                       # single
    Technical_Name: str                             # single
    CREA: str                                       # single
    ANVISA_Register: int                            # single
    Technical_Response: str                         # single
    Made_by: str                                    # multiple
    Revision: str                                   # single
    Month: str | None                               # single
    Year: int                                       # single
    Description: str                                # multiple
    Symbols: str                                    # multiple
    Specifications: str                             # multiple
    Certifications: str                             # multiple
    NBR: str                                        # multiple
    Versions: str                                   # multiple
    Working_Principle: str                          # single
    Indications: str                                # multiple
    Operation: str                                  # single
    Alarms: str                                     # multiple
    Cautions: str                                   # multiple
    Operation: str                                  # single
    Principle_Menu: str                             # single
    Setting_Menu: str                               # single
    Alarm_Menu: str                                 # single
    Passwords: str | None                           # multiple
    Buttons: str                                    # multiple
    Accessories: str                                # multiple
    Technical_Specifications: str                   # multiple
    Compatibility:str                               # single
    Electromagnetic_Compatibility: str              # single
    troubleshooting: str                            # multiple
    Attention_Alarm: str                            # single
    Emergency_Alarm: str                            # single
    Cleaning_Process: str                           # single
    Transportation: str                             # single
    Storage: str                                    # single
    Maintenance: str                                # single
    Warranty: str                                   # single

class Empresa(BaseModel):
    Name: str
    Born: int
    Country: str
    Address: str
    Portfolio: str
    Certification: str
    Email: str
    Contact: int
    Opening_Hours: str
    Technical_Support: int
    partner: str
    link: str
    social: str

    response = chat(
        message = [
            {
                'role': 'user',
                'content': ''
            }
        ],
        model = 'nuextract',
        format =
    )
