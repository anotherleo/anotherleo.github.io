let TOOLBOX_DATASET = {
    html: {
        name: "HTML5",
        type: "Linguagem de Marcação",
        description: "",
        skill_level: 4
    },
    css: {
        name: "CSS3",
        type: "Folha de estilos",
        description: "",
    },
    javascript: {
        name: "JavaScript",
        type: "Linguagem de programação",
        description: "",
    },
    bootstrap: {
        name: "Bootstrap",
        type: "Biblioteca para CSS",
        description: "",
    },
    jquery: {
        name: "jQuery",
        type: "Biblioteca para JavaScript",
        description: "",
    },
    d3js: {},
    react: {},
    vue: {},
    babel: {
        name: "Babel",
        type: "Compilador para JavaScript",
        description: "",
    },
    figma: {
        name: "Figma",
        type: "Ferramenta de Design e Prototipação",
        description: "",
    },
    git: {
        name: "Git",
        type: "Ferramenta de versionamento",
        description: "",
    },
    jest:{
        name: "Jest",
        type: "Ferramenta de Testes",
        description: "",
    }
}

const TOOLBOX_DESCRIPTION = document.querySelector('#js-toolbox-description');

function showSkillDescription(skillName) {
    let i = TOOLBOX_DATASET.skillName;

    return TOOLBOX_DESCRIPTION.innerHTML = `
        // <h3>${i.name}</h3>
        <h4>${i.type}</h4>
        <p>${i.description}</p>
    `;
}

const html = showSkillDescription('html');

console.log(html)