var mriCoord = [];
function FindPosition(oElement)
{
    console.log("In FindPos");
    if(typeof( oElement.offsetParent ) != "undefined")
    {
    for(var posX = 0, posY = 0; oElement; oElement = oElement.offsetParent)
    {
        posX += oElement.offsetLeft;
        posY += oElement.offsetTop;
    }
        return [ posX, posY ];
    }
    else
    {
        return [ oElement.x, oElement.y ];
    }
}

// function ChangeColour(PosX, PosY, myImg){
//     console.log("In Change Colour");
//     var canvas = document.createElement('canvas');
//     var context = canvas.getContext('2d');
//     var img = document.getElementById('mri');
//     // canvas.width = img.width;
//     // canvas.height = img.height;
//     context.drawImage(img, 0, 0 );
//     var imgData = context.getImageData(0, 0, img.width, img.height);

//     var i;
//     for (i = 0; i < imgData.data.length; i += 4) {
//     imgData.data[i] = 255-imgData.data[i];
//     imgData.data[i + 1] = 255-imgData.data[i + 1];
//     imgData.data[i + 2] = 255-imgData.data[i + 2];
//     imgData.data[i + 3] = 255;
//     }
//     console.log(imgData);
//     // imgData.data[0]=255;
//     // imgData.data[1]=0;
//     // imgData.data[2]=0;
//     // imgData.data[3]=255;
// }

function GetCoordinates(e){
    console.log("Making Canvas")
    var canvas = document.createElement('canvas');
    var context = canvas.getContext('2d');
    var img = document.getElementById('mri');
    canvas.width = img.width;
    canvas.height = img.height;
    context.drawImage(img, 0, 0 );

    console.log("In GETCOORD");
    console.log(myImg)  
    var PosX = 0;
    var PosY = 0;
    var ImgPos;
    ImgPos = FindPosition(myImg);
    if (!e) var e = window.event;
    if (e.pageX || e.pageY)
    {
    PosX = e.pageX;
    PosY = e.pageY;
    }
    else if (e.clientX || e.clientY)
    {
        PosX = e.clientX + document.body.scrollLeft
        + document.documentElement.scrollLeft;
        PosY = e.clientY + document.body.scrollTop
        + document.documentElement.scrollTop;
    }
    PosX = PosX - ImgPos[0];
    PosY = PosY - ImgPos[1];
    document.getElementById("x").innerHTML = PosX;
    document.getElementById("y").innerHTML = PosY;

    var imgData = context.getImageData(PosX, PosY, img.width, img.height);
    var data = imgData.data;

    // for (var y = 0; y < img.height; y++) {
    //     for (var x = 0; x < img.width; x++) {
    //         var index = (x + img.width * y) * 4;
    //         data[index+0] = data[index+2];
    //         data[index+1] = 255 - data[index+1];
    //         data[index+2] = 255 - data[index-1];
    
    //     }
    // }

    // var x = 20;
    // var y = 20;
    // data[((img.width * PosY) + PosX) * 4]=255;
    // data[((img.width * PosY) + PosX) * 4 + 1]=0;
    // data[((img.width * PosY) + PosX) * 4 + 2]=0;
    // var alpha = data[((img.width * PosY) + PosX) * 4 + 3];
    // // console.log(red);
    // console.log(data[((img.width * PosY) + PosX) * 4])


    // var i;
    // for (i = 0; i < imgData.data.length; i += 4) {
    // imgData.data[i] = 255-imgData.data[i];
    // imgData.data[i + 1] = 255-imgData.data[i + 1];
    // imgData.data[i + 2] = 255-imgData.data[i + 2];
    // imgData.data[i + 3] = 255;
    // }
    // console.log(imgData);

    // ChangeColour(PosX, PosY, myImg);
}