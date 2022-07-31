const HAMBURGUER = document.querySelector('.navbar__hamburguer');
const MENU = document.querySelector('.navbar__overlay');
const CLOSE = document.querySelector('.overlay__close');

HAMBURGUER.addEventListener('click', () => {
    MENU.classList.toggle('open');
})

CLOSE.addEventListener('click', () => {
    MENU.classList.toggle('open');
})