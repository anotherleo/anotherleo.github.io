
/* Controle do Modal */
function startModal(modalID) {
    const modal = document.getElementById(modalID);
    modal.classList.add('show');

    /* função para fechar ao clicar fora OU no botão OU no ícone de fechar da FontAwesome */
    modal.addEventListener('click', (e) => {
        if (e.target.id == modalID || e.target.className == 'modal-close' || e.target.className == 'fas fa-times fa-lg') {
            modal.classList.remove('show');
        }
    })
}

/* Variável de cada modal */
const projectOne = document.getElementById('project-one');
projectOne.addEventListener('click', function () {
    startModal('modal-project-one');
})

const projectTwo = document.getElementById('project-two');
projectTwo.addEventListener('click', () => startModal('modal-project-two'));
/* '() =>' é recurso do ES6, equivale a 'function ()' {} */

const projectThree = document.getElementById('project-three');
projectThree.addEventListener('click', () => startModal('modal-project-three'));

const projectFour = document.getElementById('project-four');
projectFour.addEventListener('click', () => startModal('modal-project-four'));