var mriCoord = [];
var ctCoord = [];
function FindPosition(oElement) {
    if (typeof (oElement.offsetParent) != "undefined") {
        for (var posX = 0, posY = 0; oElement; oElement = oElement.offsetParent) {
            posX += oElement.offsetLeft;
            posY += oElement.offsetTop;
        }
        return [posX, posY];
    }
    else {
        return [oElement.x, oElement.y];
    }
}

function GetCoordinatesMri(e) {
    if (mriCoord.length < points) {
        var PosX = 0;
        var PosY = 0;
        var ImgPos;
        ImgPos = FindPosition(myImgMri);
        if (!e) var e = window.event;
        if (e.pageX || e.pageY) {
            PosX = e.pageX;
            PosY = e.pageY;
        }
        else if (e.clientX || e.clientY) {
            PosX = e.clientX + document.body.scrollLeft
                + document.documentElement.scrollLeft;
            PosY = e.clientY + document.body.scrollTop
                + document.documentElement.scrollTop;
        }
        PosX = PosX - ImgPos[0];
        PosY = PosY - ImgPos[1];
        mriCoord.push([PosX, PosY]);
        document.getElementById("mriX").innerHTML = PosX;
        document.getElementById("mriY").innerHTML = PosY;

        var imgData = context.getImageData(PosX, PosY, img.width, img.height);
        var data = imgData.data;
    } else {
        alert("Can't exceed number of points");
    }
}

function GetCoordinatesCt(e) {

    if (ctCoord.length < points) {
        console.log(myImgCt)
        var PosX = 0;
        var PosY = 0;
        var ImgPos;
        ImgPos = FindPosition(myImgCt);
        if (!e) var e = window.event;
        if (e.pageX || e.pageY) {
            PosX = e.pageX;
            PosY = e.pageY;
        }
        else if (e.clientX || e.clientY) {
            PosX = e.clientX + document.body.scrollLeft
                + document.documentElement.scrollLeft;
            PosY = e.clientY + document.body.scrollTop
                + document.documentElement.scrollTop;
        }
        PosX = PosX - ImgPos[0];
        PosY = PosY - ImgPos[1];


        ctCoord.push([PosX, PosY]);
        document.getElementById("ctX").innerHTML = PosX;
        document.getElementById("ctY").innerHTML = PosY;

        var imgData = context.getImageData(PosX, PosY, img.width, img.height);
        var data = imgData.data;
    } else {
        alert("Can't exceed number of points");
    }
}

function sendParameters() {
    console.log("put method")

    $.post('/register', {
        mriCoord: JSON.stringify(mriCoord),
        ctCoord: JSON.stringify(ctCoord),
    }, function (res) {
        location.href = '/registerimage';
    });
}