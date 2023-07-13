css = '''
<style>
body {
    background-color: #000000;  /* Color de fondo para todo el cuerpo de la página */
    color: #ffffff;  /* Color de texto general para todo el cuerpo de la página */
}
.reportview-container {
    max-width: 60%;
    margin: auto;
    background-color: #262626;  /* Cambia el color de fondo del contenedor de la aplicación Streamlit */
    color: #ffffff;  /* Cambia el color de texto del contenedor de la aplicación Streamlit */
    padding: 10px;  /* Agrega algo de espacio alrededor del contenedor de la aplicación Streamlit */
}
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 50%;
  padding: 0 1.5rem;
  color: #fff;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTu6i8CWYY8EJEeYKu_jZtVlFpFFgLRoO03wA&usqp=CAU" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/512/186/186313.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
