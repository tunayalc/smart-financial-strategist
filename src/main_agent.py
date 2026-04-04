from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent
from agent_tools import get_portfolio_tool, get_justification_tool

def run_agent():
    print("Ajan başlatılıyor...")

    llm = ChatOllama(model="llama3", base_url="http://ollama_core:11434")
    tools = [get_portfolio_tool, get_justification_tool]

    prompt_template = """
    Sen bir finansal stratejistsin. 
    Kullanıcının girdisine göre risk profilini belirle, portföy oluştur ve her varlık için açıklama yap.
    Raporu markdown formatında döndür.

    Kullanıcı: {input}
    Ara Adımlar: {agent_scratchpad}
    """
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    user_goal = "5 yıl içinde bir ev peşinatı biriktirmek istiyorum. Çok fazla risk almak istemiyorum."
    print(f"Kullanıcı Hedefi: {user_goal}")

    result = agent_executor.invoke({"input": user_goal})
    
    print("Ajan Çıktısı:")
    print(result["output"])

if __name__ == "__main__":
    run_agent()