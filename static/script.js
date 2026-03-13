function openTab(tab){

let tabs = document.getElementsByClassName("tabcontent");

for(let i=0;i<tabs.length;i++){
tabs[i].style.display="none";
}

document.getElementById(tab).style.display="block";

}

async function predict(){

let input = document.getElementById("inputFeatures").value;

let features = input.split(",").map(Number);

let response = await fetch("/predict",{

method:"POST",

headers:{
"Content-Type":"application/json"
},

body:JSON.stringify(features)

});

let data = await response.json();

document.getElementById("result").innerHTML =
"Prediction: "+data.prediction;

}