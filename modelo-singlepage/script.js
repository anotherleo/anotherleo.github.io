
/* Controle do Modal */
function startModal(modalID) {
    const modal = document.getElementById(modalID);
    modal.classList.add('show');
}

const projectOne = document.getElementById('project-one');
projectOne.addEventListener('click', function () {
    startModal('modal-project-one');
})

const projectTwo = document.getElementById('project-two');
projectTwo.addEventListener('click', () => startModal('modal-project-two'));
/* '() =>' é recurso do ES6, equivale a 'function ()' {} */

const projectThree = document.getElementById('project-three');
projectThree.addEventListener('click', () => startModal('modal-project-three'));