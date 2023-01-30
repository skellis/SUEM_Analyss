import numpy as np
import glob
from numpy.fft import fft2, ifft2, fftshift
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import path
import matplotlib.ticker as ticker
import abel
import os, sys
from PIL import Image, ImageChops, ImageDraw,ImageSequence,ImageFont
import cv2 as cv
import time
from time import gmtime, strftime
import math
import pandas as pd
import itertools
import scipy.stats
from scipy import linalg
from scipy import ndimage
from scipy import special
from scipy.integrate import odeint
import scipy.optimize as opt
from scipy.optimize import curve_fit
from scipy.stats.distributions import t
#matplotlib.rcParams['text.usetex'] = True

try:
	import scipy.ndimage.interpolation as ndii
except ImportError:
	import ndimage.interpolation as ndii
"""
.. module: suemimageutilities.py
   :platform: macOSMojave
.. moduleauthor:: Scott R. Ellis <srellis@sandia.gov>

Last organized 2020.04.07 in 20200110-VONWonAuStripe
Note timepoints must have decimals d10.0

This is a suit of modules for streamlining the data processing workflow for a scaning ultrafast microscope at Sandia Natinal Labs.


Relevant package install commands. Make sure get-pip.py in in cd
get-pip.py can be found at 
https://github.com/pypa/get-pip


python3 get-pip.py
python3 -m pip install -U numpy --user
python3 -m pip install -U glob --user
python3 -m pip install -U matplotlib --user
python3 -m pip install -U PIP --user
python3 -m pip install -U openpyxl -
python-scitools

You also may want Mactex
Mactex can be found at 
http://www.tug.org/mactex/


SECTION 1 Initiialization and File Handling
		getfilenames
		generatefilename
		getfileprefix
		coord2um
		energyinchamber
		energyofphoton
		getpositionaxis
		loadtimepoints
		mm2psdelay
		ps2mmdelay
		formatdecimalstring
		createtimepoints
		addcaption
		addcaptionmulti
		getfilenames
		generategif
		converttojpeg
		createlatexfile
		runlatexfile
SECTION 2 For batch data loading and processing
		processlongdatamulti
		loadlongdata
		getcutpoints
		getdatamat
		assembletyxmat
		loadtyxmat
SECTION 3 Plotting & Image Generationg		
		createimage
		createimagemulti
		plotkinetictracemulti
		plotROItracemulti
		plotkinetictrace
		plotlinecut
SECTION 4 Subtracting, Stretching, Rotating, Shifting,Phase Rotation
		getstretchfactor
		getstretchfactor2
		getstretchfactormulti
		stretchdata
		stretchdatamulti
		adjustdatamulti
		clickimagecoords
		getcoordsfromclickmulti
		rotatephasemat
		voltage2radians
		radians2voltage
		rtheta2x
		rtheta2y
		findphaseofminstd
		getphasefactormulti
		getphasefactorfromclick
		flipphasefactor
		rotatephasenmulti
		rotatedatamulti
		getshiftvector
		getshiftvectorfromclickmulti
		getshiftvectormulti
		shiftdata
		shiftdatamulti
		similarity
		logpolar
SECTION 5 Masking, Averaging, Regions of Interest			
		createmask
		createannularmask
		applycircularmaskmulti
		createannularellipsemask
		getellipseparams
		appyannularmaskmulti
		averageROI
		averageROImulti
		createellipseparametric
		createellipseparametricmulti
		subtractkinetictracemulti
		linecut
		linecutmulti
SECTION 6 Fitting

		fitlinecutmulti
		fittimeseries_global
		fittimeseries
		ccrwrs
		ccrwrs_conv_lin
		ccrwrs_conv_linb
		ccrwrs_conv
		ccrwrs_conv_interp
		ccrwrs_conv_interpb
		exp_conv_interp
		dblexp_conv_interp
		exp_conv_interpcccrwrs_conv_interp_global
		cumulative
		dgaussdx
		cumulativeplusdgaussdx
		gaussplusdgaussdx
		trimData
		createbounds
		fit2dring
		simulate2Dfunction
		gauss2d
		gauss2drot
		superguass2drot
		superguass2drotoffset
		doublesuperguass2drot
		doublesuperguass2drotoffset
		multisuperguass2drotoffset
		doublegauss2drot
		twoD_Gaussian
		multisuperguass2drotoffset
		multisuperguass2drotoffset2
		multisuperguass2drotoffset3
		multisuperguass2drotoffset4
		fluenceresponse0
		beamprofilefluence
		beamprofilefluence_rot
		getprofilemat
		simulatepowerseries
SECTION 99 Useful Functions		
		lin_equ
		lin_coords
		angle
		vectorize
		dotproduct
		length
		image2array
		array2image
		grayscaleimage
		line_intersection
		rotateimage
		translateimage
		removeinstrumentresponse
		clockwiseangle_and_distance
		PolygonArea
		applylowpassfilter
		points_in_circle
		highpass
		mean_confidence_interval
"""



"""
========================================================================================================
SECTION 1 Initiialization, Constants, File Handling
========================================================================================================

#Example code for processing SUEM time series data.
import pysuem as ps
#Set up instrument parameters
dp=ps.loadtimepoints("delaypoints.txt")
tp=ps.formatdecimalstring(ps.loadtimepoints("timepoints.txt"),0)
samplelabel0='SiO\\textsubscript{2} P-Type Wet Thermal Oxide'
searchstring0="wet1umsio2_CA_pump35mW_l2_160_fd2_tc300us_116_mag800x"
destinationfileprefix="wet1umsio2"
power0=26
av0=15
mag0=800
wd0=14
lambda0=515
polarizer0=160
reprate0=2
fdiv0=2
#Run operating commands
if 1:
	#get the file list of the 2 MHz image
	fl02MHz=ps.getfilenames(searchstring0+"_p0*txt_ch3stretched",verbose=1,suffix='.jpg')
	#create image files with added captions
	ps.addcaptionmulti(fl02MHz,searchstring0+"_2MHz",wd=wd0,mag=mag0,av=av0)
	#get the file list of newly created images
	fl02MHzcap=ps.getfilenames(searchstring0+"_2MHz",verbose=1,delimiter="_t",suffix='.tif')
	#convert tifs to jpgs
	ps.converttojpeg(filelist=fl02MHzcap,verbose=1)
if 1:
	#get the file list of the amplitde images R
	fl0R=ps.getfilenames(searchstring0,index=dp,delimiter="_d",suffix=".txt_ch1.jpg",verbose=1)
	#creat image files with added captions
	ps.addcaptionmulti(fl0R,searchstring0+"_R0",tp,prefix="_t",timepoints=tp,power=power0,wd=wd0,mag=mag0,av=av0)
	#get the file list of newly created images
	fl0Rcap=ps.getfilenames(searchstring0+"_R0",index=tp,verbose=1,delimiter="_t",suffix='.tif')
	if 0:
		#generate a gif move of the newly captioned files!!!
		ps.generategif(fl0Rcap,destinationfileprefix+'_R0.gif',resizeratio=6,duration=1500)
	#convert tifs to jpgs
	ps.converttojpeg(filelist=fl0Rcap,verbose=1)
if 1:
	#get the file list of the phase images Theta
	fl0Theta=ps.getfilenames(searchstring0,index=dp,delimiter="_d",suffix=".txt_ch2.jpg",verbose=1)
	#creat image files with added captions
	ps.addcaptionmulti(fl0Theta,searchstring0+"_Theta0",tp,prefix="_t",timepoints=tp,power=power0,wd=wd0,mag=mag0,av=av0)
	#get the file list of newly created images
	fl0Thetacap=ps.getfilenames(searchstring0+"_Theta0",index=tp,verbose=1,delimiter="_t",suffix='.tif')
	if 0:
		#generate a gif move of the newly captioned files!!!
		ps.generategif(fl0Thetacap,destinationfileprefix+'_Theta0.gif',resizeratio=6,duration=1500)
	#convert tifs to jpgs
	ps.converttojpeg(filelist=fl0Thetacap,verbose=1)
if 1:
	#getfile names of jpgs
	fl02MHzjpeg=ps.getfilenames(searchstring0+"_2MHz*",verbose=1,delimiter="_t",suffix='.jpeg')
	fl0Rjpeg=ps.getfilenames(searchstring0+"_R0",index=tp,verbose=1,delimiter="_t",suffix='.jpeg')
	fl0Thetajpeg=ps.getfilenames(searchstring0+"_Theta0",index=tp,verbose=1,delimiter="_t",suffix='.jpeg')
	ps.createlatexfile(destinationfilename=destinationfileprefix+'_Timeseries.tex',samplelabel='SiO\\textsubscript{2} P-Type Wet Thermal Oxide',sampleparams=str(av0)+' kV, Mag '+str(mag0)+'x, '+str(power0)+' mW, '+str(lambda0)+' nm, Polarizer '+str(polarizer0)+'\\textsuperscript{o}, Working Distance +'+str(wd0)+' mm, '+str(reprate0)+' MHz, Frequency Dividing '+str(fdiv0),rfilelist=fl0Rjpeg,thetafilelist=fl0Thetajpeg,twomhzfilelist=fl02MHzjpeg)
	ps.runlatexfile(filename=destinationfileprefix+'_Timeseries.tex')
"""

def getfilenames(basename, index=None,extrastring="",delimiter="_d",suffix="",integerindex=0,exactmatch=0,verbose=0,oldvalue="",newvalue="x"):
	"""
	a module for getting a list of image files which match the match strong specifier "basename" 
	and optionally have the timepoint in the file name.

	:param string basename: A specifier string to match the base of the files you want to collect should include astrix.
	:param list float: index: the collection of index which are included in the file name.
	:param string: extrastring: a utility match string which can be left blank.
	:param string: delimiter: delimiter preceeding index. default "d_"
	:param string: suffix: match string proceeding index
	:param boolean: print results for debugging

	Example
	************************
	 fl=getfilenames("1umsio2VONWcurrentamphg_pump32mW_l2_160_1MHz_tc300us_116_mag800x*",index=mp,prefix="MP_",suffix="*_d0.txt_ch2stretched.jpg",verbose=0)
	"""
	searchstring=""
	filelist=[]
	filenames=[]
	if(index is not None):
		for t in index:
			if(integerindex):
				if(exactmatch):
					searchstring=basename+extrastring+delimiter+str(int(t))+suffix
				else:
					searchstring=basename+"*"+extrastring+"*"+delimiter+str(int(t))+"*"+suffix
			else:
				if(exactmatch):
					searchstring=basename+extrastring+delimiter+str(t)+suffix
				else:
					searchstring=basename+"*"+extrastring+"*"+delimiter+str(t)+"*"+suffix
			if oldvalue is not "":
				searchstring =searchstring.replace(oldvalue,newvalue,(searchstring.count(oldvalue)-1))
			filenames = glob.glob(searchstring)
			if verbose:
				print(searchstring)
				print(filenames)
			filelist+=filenames
	else:
		if(exactmatch):
			searchstring=basename+extrastring+suffix
		else:
			searchstring=basename+"*"+extrastring+suffix
		if oldvalue is not "":
			searchstring =searchstring.replace(oldvalue,newvalue,(searchstring.count(oldvalue)-1))
		filenames = glob.glob(searchstring)
		filelist=filenames
		if verbose:
			print(searchstring)
			print(filenames)
	return filelist

def getfilenamesdoubleintindex(base,xindex, yindex,suffix=".txt",verbose=0):
	xindex2=["0"+str(a) if a<10 else str(a) for a  in xindex]
	yindex2=["0"+str(a) if a<10 else str(a) for a  in yindex]
	fl=[]
	for xi in xindex2:
		for yi in yindex2:
			searchstring=base+"_X"+xi+"_Y"+yi+suffix
			filenames=glob.glob(searchstring)
			if filenames !=[]:
				if verbose:
					print(filenames)
				fl+=filenames
	return fl

def generatefilename(filename,suffix="",extension="txt",verbose=0):
	"""
	A module for adding a suffix to a filename before the file type extension
	If no file '.' is found an extension is added by default.
	:param string filename: file name to be modified
	:param string suffix: string to be added before extension
	:param string extension: what extension should be used
	"""
	splitname=filename.split(".")
	delimiter="."
	#no periods in file name
	if(len(splitname)==1):
		newfilename= splitname[0]+suffix+delimiter+extension
	#one period and it indicates a file extenstion
	#only replace the extension if one is not explicitly given besides .txt
	elif(len(splitname[-1])==3):
		#when extension is default i.e. not given use the extension in filename
		if(extension=="txt"):
			extension=splitname[-1]
		newfilename= filename[:-4]+suffix+delimiter+extension
	#period does not delimit an extension so lets add one.
	elif(len(splitname[-1])!=3):
		newfilename= filename+suffix+delimiter+extension
	if(verbose):
		print("The old filename:", filename)
		print("The new filename: ",newfilename)
	return newfilename

def generatefilenamemulti(filelist,suffix="",extension="txt",verbose=0):
	"""
	A module for adding a suffix to a filename before the file type extension
	If no file '.' is found an extension is added by default.
	:param string filename: file name to be modified
	:param string suffix: string to be added before extension
	:param string extension: what extension should be used
	"""
	fl=[]
	for f in filelist:
		fn_new=''
		fn_new=generatefilename(f,suffix=suffix,extension=extension,verbose=verbose)
		fl.append(fn_new)
	return fl

def cleanfilename(filename,character=".",replacementchar="x"):
	"""
	a module for removing all but the last of a specific character i.e. periods from a filename
	"""
	counter = 0
	for i in filename: 
		if i == '.': 
			counter = counter +1 
	return filename.replace(character, replacementchar,counter)

def cleanfilelist(filelist,character=".",replacementchar="x"):
	cleanfilelist=[]
	for f in filelist:
		cfn=cleanfilename(f,character=character,replacementchar=replacementchar)
		cleanfilelist.append(cfn)
	return cleanfilelist

def renamefile(oldfilename,newfilename):
	np.savetxt(newfilename,np.loadtxt(oldfilename))
	return

def renamefile_multi(oldfilelist,newfilelist):
	for i in range(len(oldfilelist)):
		renamefile(oldfilelist[i],newfilelist[i])
	return

def getvaluefromf(f, delimiter1,delimiter2):
	return f.split(delimiter1)[len(f.split(delimiter1)) -1 ].split(delimiter2)[0]

def getvaluefromfmulti(fl,delimiter1=" ",delimiter2=" ",tofloat=0,sigfigs=0):
	fvalues=[f.split(delimiter1)[len(f.split(delimiter1)) -1 ].split(delimiter2)[0] for f in fl]
	if tofloat>0:
		fvalues_flt=[float(n) for n in fvalues]
		return fvalues_flt
	return fvalues

def getfileprefix(filename,delimiter="_p",verbose=0):
	"""
	A module that stripe the portion of a string that occurs before delimiter"""
	splitname=filename.split(delimiter)
	if(verbose):
		print(filename[0:filename.rfind(delimiter)])
	return filename[0:filename.rfind(delimiter)]

def buildmagdictionary():
	"""
	A module for building a dictionary which where the entries are the magnifications and the definitions are the 
	fraction of the image horizontally the scale bar makes up and the caption.
	Example
	************************
	SD=buildscalebardictionary()
	"""
	magDict={66:{'scaling':0.16958*3/2,'label':'1 mm','value':1000./0.16958},
	100:{'scaling':0.16958,'label':'500 μm','value':500./0.16958},
	150:{'scaling':0.16958*1.5,'label':'500 μm','value':500./(0.16958*1.5)},
	200:{'scaling':0.16958*4/5,'label':'200 μm','value':200./(0.16958*4./5.)},
	350:{'scaling':0.16958*14/10,'label':'200 μm','value':200./(0.16958*14./10.)},
	500:{'scaling':0.16958,'label':'100 μm','value':100./(0.16958)},
	650:{'scaling':0.16958*6.5/5,'label':'100 μm','value':100./(0.16958*6.5/5)},
	800:{'scaling':0.16958*4/5,'label':'50 μm','value':50./(0.16958*4/5)},
	1000:{'scaling':0.16958,'label':'50 μm','value':50./(0.16958)},
	1200:{'scaling':0.16958*1.2,'label':'50 μm','value':50./(0.16958*1.2)},
	1500:{'scaling':0.16958*1.5,'label':'50 μm','value':50./(0.16958*1.5)},
	2000:{'scaling':0.169279*4/5,'label':'20 μm','value':20./(0.16958*4/5)},
	2500:{'scaling':0.169279,'label':'20 μm','value':20./(0.16958)},
	5000:{'scaling':0.169279,'label':'10 μm','value':10./(0.16958)},
	6500:{'scaling':0.169279*6.5/5,'label':'10 μm','value':10./(0.16958*6.5/5)},
	10000:{'scaling':0.169279,'label':'5 μm','value':5./(0.16958)},
	12000:{'scaling':0.169279*1.2,'label':'5 μm','value':5./(0.16958*1.2)},
	25000:{'scaling':0.169279,'label':'2 μm','value':2./(0.16958)},
	20000:{'scaling':0.169279*4/5,'label':'2 μm','value':2./(0.16958*4/5)},
	50000:{'scaling':0.169279,'label':'1 μm','value':1./(0.16958)},
	11111:{'scaling':0.46/2.28,'label':'50 μm','value':1./(0.46/2.28)}
	}
	return magDict

def pixel2um(pixel,mag,xpixel=738,ypixel=485):
	magDict=buildmagdictionary()
	xaxislength=magDict[mag]['value']
	return np.multiply(pixel,xaxislength/xpixel)

def coord2um(coord,mag,xpixel=738,ypixel=485):
	return pixel2um(np.array([coord[0]-xpixel/2,coord[1]-ypixel/2]),mag,xpixel=738,ypixel=485)



def frenelreflectivity(n2,thetai=54,n1=1,polariztion='u'):
	"""
	Return the reflectivity coefficient R from the frenel equations from the index of refraction at an interfact for a given angle of incidence.
	Example code
	import pysuem as ps
	print(ps.frenelreflectivity(4.1))
	"""
	thetai=thetai*np.pi/180
	Rs=np.absolute(((n1*np.cos(thetai)-n2*(1-(n1/n2*np.sin(thetai))**2)**0.5)/(n1*np.cos(thetai)+n2*(1-(n1/n2*np.sin(thetai))**2)**0.5)))**2
	Rp=np.absolute(((n1*(1-(n1/n2*np.sin(thetai))**2)**0.5-n2*np.cos(thetai))/(n1*(1-(n1/n2*np.sin(thetai))**2)**0.5+n2*np.cos(thetai))))**2
	Ru=(Rs+Rp)/2
	if polariztion=='u':
		return Ru
	elif polariztion=='s':
		return Rs
	else:
		return Rp
	return

def energyinchamber(power,mirrorlosses=11./65.,materialreflectivity=0,reprate=1E6):
	"""
	Takes mW returns Joules
	"""
	return power/1000*(1-materialreflectivity)*mirrorlosses/reprate

def fluenceinchamber(power,materialreflectivity=0,reprate=1E6,wx=42,wy=24.5):
	"""
	Takes mW returns mJ/cm^3
	"""
	return energyinchamber(power,materialreflectivity=materialreflectivity,reprate=reprate)*2/np.pi/(wx*10**-4)/(wy*10**-4)*1000

def powerbeforechamber(fluence,materialreflectivity=0,reprate=1E6,wx=42,wy=24.5):
	"""
	Takes mJ/cm^3 returns mW
	"""
	return fluence/2*np.pi*(wx*10**-4)*(wy*10**-4)/1000*1000/(1-materialreflectivity)*2100/962*reprate

def energyofphoton(l=0.515):
	"""
	l -> lambda wavelength in units of microns
	"""
	return (1.98644568*10**-25)/(l/10**6)

def getpositionaxis(mag, xpixel=738,ypixel=485,origintopleft=0):
	#comment here
	"""
	block comment
	"""
	magDict=buildmagdictionary()
	xaxislength=magDict[mag]['value']
	yaxislength=xaxislength*485./738.
	if origintopleft:
		shiftx=np.linspace(0, xaxislength, num=xpixel)
		shifty=np.linspace(0, yaxislength, num=ypixel)
	else:
		shiftx=np.linspace(-xaxislength/2, xaxislength/2, num=xpixel)
		shifty=np.linspace(-yaxislength/2, yaxislength/2, num=ypixel)
	return shiftx, shifty

def createtimepoints(delaypoints,destination='timepoints.txt',delayzero=0.0):
	tp0=formatdecimalstring(mm2psdelay(delaypoints,delayzero=delayzero),0)
	with open(destination, 'w+') as f:
		for item in tp0:
			f.write(item+"\n")
		f.close()
	return tp0

def loadtimepoints(f,verbose=0):
	"""
	:param string f: file name to
	:param boolean: verbose: print the array 
	Used subsequently in identifying file names.
	
	Example
	************************
	movepump=loadtimepoints("MP.txt")
	timepoints=loadtimepoints("timepoints.txt")
	"""
	timepoints=np.loadtxt(f, unpack=True)
	if verbose:
		print(timepoints)
	return timepoints

def mm2psdelay(mm,delayzero=0.0):
	"""
	A module for converting mm of delay on our 4 pass optical setup to ps of time delay.
	:param float: mm: value of delay stage in mm
	2.99792458*10**(-2) mm/ps
	"""
	return -(mm-delayzero)*4/(2.99792458*10**(-1))

def ps2mmdelay(ps,timezero=0.0):
	"""
	A module for converting ps of delay on our 4 pass optical setup to mm.
	:param float: ps: time of delay between light and electrons ps
	2.99792458*10**(-2) mm/ps
	"""
	return -(ps-timezero)/4*(2.99792458*10**(-1))

def formatdecimalstring(val,decimals,flipzeros=0,replacementchar="."):
	"""
	A module for converting floats or np.ndarrays to strings or lists of strings with a determined number of decimals places


	Examples
	*********************************
	dp=ps.loadtimepoints("delaypoints.txt")
	tp0=ps.mm2psdelay(dp)
	print(ps.decimalmanagestring(tp0,1))
	>>['73384.1', '26685.1', '13342.6', '6671.3', '2668.5', '1334.3', '667.1', '266.9', '0.0', '-266.9', '-400.3', '-533.7', '-600.4', '-667.1', '-733.8', '-800.6', '-867.3', '-934.0', '-1067.4', 
	>>'-1334.3', '-1601.1', '-2001.4', '-2668.5', '-6671.3', '-13342.6', '-26685.1', '-73384.1']
	print(ps.decimalmanagestring(3.1415,1))
	>>3.1
	"""
	format="%."+str(decimals)+"f"
	if(isinstance(val, float)):
		if(decimals==0):
			s=str(val).split(".")[0]
		elif(val is 0.0 and flipzeros):
			s=-0.0
		else:
			s=format % val
		return s
	elif(isinstance(val, np.ndarray)):
		t=[]
		for bb in val:
			if(decimals==0):
				t.append(str(bb).split(".")[0].replace(".",replacementchar,1))
			elif(bb == 0 and flipzeros):
				t.append('-0.0'.replace(".",replacementchar,1))
			else:
				t.append((format % bb).replace(".",replacementchar,1))
		return t
	return

def addcaption(filename,destination,timepoint=-0.333333,power=-0.222222,mag=0,wd=0,av=0,timecaption=" ps",powercaption=" mJ/cm^2",wdcaption="WD =		 mm",avcaption="AV =	  KV",rgba=(200,0,0,255),show=0,save=0,getcaptionimage=0,decimals=3):
	"""
	A module for creating an image (SEM) file with a caption with labels like timepoints powers, magnifications, 
	working distances, accelerating voltage. 
	
	:param string filename: file name to load and add caption to.
	:param string: destination: path where image will be saved
	:param float: timepoint: optional value to add to caption. time delay (ps) 
	:param float: power: optional value to add to caption. power (mW)
	:param float: mag: optional value to add to caption. magnification e.g. 1000x
	:param float: wd: optional value to add to caption. working distance (mm)
	:param float: av: optional value to add to caption. Accelerating voltage (av)
	:param tuple: rgba: optional value to designate color of the captino default is a dark red.
	:param boolean: show: option to show the image which has been created
	:param boolean: save: option to save the image which has been created
	:param boolean: getcaptionimage: option to return the image which has been created

	:returns: Optionally returns the image

	 .. seealso:: addcaptionmulti:  Often used in batch with add caption multi

	Example
	************************
	addcaption("1umsio2VONWcurrentamphg_pump32mW_l2_160_1MHz_tc300us_116_mag800x-MP_d0.txt_ch1stretched.jpeg",destination,timepoints,32,1000,14,30,rgba=(255,255,255,255),save=1)
	"""
	lineyfrac=.98
	im=Image.open(filename).convert('RGBA')
	imwidth, imheight = im.size 
	draw = ImageDraw.Draw(im)
	decimalhandling='{:.'+str(decimals)+'f}'
	# font = ImageFont.truetype(<font-file>, <font-size>)
	#
	if(imwidth<800 and imwidth>700):
		fontsize=41
		linewidth=10
	else:
		fontsize=600
		linewidth=100

	font = ImageFont.truetype("fonts/arial.ttf", fontsize)
	fontsymbol = ImageFont.truetype("fonts/symbol.ttf", fontsize)
	caption = Image.new('RGBA', im.size, 0)
	d = ImageDraw.Draw(caption)
	# draw text
	print(timepoint)
	if(timepoint !=-0.333333):
		w, h = draw.textsize('{:.0f}'.format(int(timepoint)) ,font=font)
		d.text((0.125*imwidth-w, 0.02*imheight), '{:.0f}'.format(int(timepoint)), font=font, fill=rgba)
		d.text((0.13*imwidth, 0.02*imheight), timecaption, font=font, fill=rgba)
	if(power !=-0.222222):
		w2, h2 = draw.textsize(decimalhandling.format(power) ,font=font)
		d.text((0.5*imwidth-w2, 0.02*imheight), decimalhandling.format(power), font=font, fill=rgba)
		d.text((0.51*imwidth, 0.02*imheight), powercaption, font=font, fill=rgba)
	# draw.text((x, y),"Sample Text",(r,g,b))
	if(mag>0.0):
		magDict=buildmagdictionary()
		maglabel=magDict[mag]['label']
		magscaling=magDict[mag]['scaling']
		d.text(((.02)*imwidth, imheight*87/100), maglabel, font=font, fill=rgba)
		d.line((0.02*imwidth,imheight*lineyfrac,(.02+magscaling)*imwidth, imheight*lineyfrac) , fill=rgba,width=linewidth)
	if(wd>0):
		w3, h3 = draw.textsize(str(wd) ,font=font)
		d.text((.24*imwidth, imheight*87/100), wdcaption, font=font, fill=rgba)
		d.text((.5*imwidth-w3, imheight*87/100), str(wd), font=font, fill=rgba)
	if(av>0):
		w4, h4 = draw.textsize(str(av) ,font=font)
		d.text((.92*imwidth-w4, imheight*87/100), str(int(av)), font=font, fill=rgba)
		d.text((.68*imwidth, imheight*87/100), avcaption, font=font, fill=rgba)
	out = Image.alpha_composite(im, caption)
	if save:
		out.save(destination)
	if show:
		out.show()
	if getcaptionimage:
		return out
	else:
		return

def	addcaptionmulti(filelist,destinationbase,index=None,prefix="_",timepoints=None,power=None,mag=None,wd=None,av=None,timecaption=" ps",powercaption=" mJ/cm^2",wdcaption="WD =		 mm",avcaption="AV =	   KV",rgba=(200,0,0,255),integerindex=0,decimals=3):
	"""
	A module for creating a batch of image (SEM) file with a caption with labels like timepoints powers, magnifications, 
	working distances, accelerating voltage. 
	
	:param string filelist: list of file names to load and add caption to.
	:param string: destinationbase: base from which file names are formed
	:param list float: index: optional list of  values to add to caption. time delay (ps) 
	:param list float: power: optional value to add to caption. power (mW)
	:param list float: mag: optional value to add to caption. magnification e.g. 1000x
	:param list float: wd: optional value to add to caption. working distance (mm)
	:param list float: av: optional value to add to caption. Accelerating voltage (av)
	:param tuple: rgba: optional value to designate color of the captino default is a dark red.


	 .. seealso:: getfilenames:  For generating fileliest
	 .. seealso:: addcaption:  For generating adding a single caption

	Example
	************************
	addcaption("1umsio2VONWcurrentamphg_pump32mW_l2_160_1MHz_tc300us_116_mag800x-MP_d0.txt_ch1stretched.jpeg",destination,index,32,1000,14,30,rgba=(255,255,255,255),save=1)
	"""
	destination=""
	if(index is None):
		index=np.ones(len(filelist))*-0.111111
	elif(isinstance(index, int)):
		index=np.ones(len(filelist))*index
	
	if(power is None):
		power=np.ones(len(filelist))*-0.222222
	elif(isinstance(power, int) or isinstance(power, float)):
		power=np.ones(len(filelist))*power
	print(timepoints)
	if(timepoints is None):
		timepoints=np.ones(len(filelist))*-0.333333
	elif(isinstance(timepoints, int) or isinstance(timepoints, float)):
		timepoints=np.ones(len(filelist))*timepoints
	
	if(mag is None):
		mag=np.zeros(len(filelist))
	elif(isinstance(mag, int) or isinstance(mag, float)):
		mag=np.ones(len(filelist))*mag
	
	if(wd is None):
		wd=np.zeros(len(filelist))
	elif(isinstance(wd, int) or isinstance(wd, float)):
		wd=np.ones(len(filelist))*wd
	
	if(av is None):
		av=np.zeros(len(filelist))
	elif(isinstance(av, int) or isinstance(av, float)):
		av=np.ones(len(filelist))*av
	i=0
	for f in filelist:
		destination =generatefilename(f,suffix="_cap",extension="tif",verbose=0)
		addcaption(f,destination,timepoints[i],power[i],mag[i],wd[i],av[i],timecaption=timecaption,powercaption=powercaption,wdcaption=wdcaption,avcaption=avcaption,rgba=rgba,save=1,decimals=decimals)
		i+=1
	return

def generategif(filelist,destinationpath,resizeratio=0,duration=1000):
	"""
	Note pillow has 8 bit pixel depth. For HD gifs use PicGif Lite form the apple store
	A module for generating a gif from a list of image files which must appear in the current directory.
	a destination where the gif is to be created.

	:param list string: filelist: A list of files to load
	:param string: destinationpath: filepatha and file name where gif is to be written. should end in '.gif'
	:param string: extrastring: a utility match string which can be left blank.
	:param float: resizeratio: decrease gif resolution by a factor or resizeratio to managable size.
	:param float: duration: how slow should the gif be?
 
	"""
	filelist.insert(0,filelist[0])
	resize=0
	img, *imgs = [Image.open(f) for f in filelist]
	if(resizeratio is not 0):
		img=imgs[0].resize((imgs[0].size[0]//resizeratio,imgs[0].size[1]//resizeratio),Image.ANTIALIAS)
		imgs=[tmp.resize((imgs[0].size[0]//resizeratio,imgs[0].size[1]//resizeratio),Image.ANTIALIAS) for tmp in imgs]
	img.save(fp=destinationpath, format='GIF', 
		append_images=imgs,save_all=True, optimize=False, duration=duration, loop=0)
	return

def converttojpeg(filelist=[], verbose=0,quality=90,replacementchar="."):
	"""
	Converts all files with .tif in the file name to a .jpg file.
	taken from https://stackoverflow.com/questions/28870504/converting-tiff-to-jpeg-in-python


	:param list string: filelist: A list of files to convert if list is empty this program find all the files that end in tif or bmp and converts them
	:param boolean: verbose: print the results

	"""
	if(filelist is []):
		filelist=os.listdir("./")
	for infile in filelist:
		if(verbose):
			print("file : " + infile)
		if infile[-4:] == ".tif" or infile[-4:] == ".bmp" :
			# print "is tif or bmp"
			outfile = infile[:-4].replace(".",replacementchar) + ".jpeg"
			im = Image.open(infile)
			if(verbose):
				print("new filename : " + outfile)
			out = im.convert("RGB")
			out.save(outfile, "JPEG", quality=quality)

def createlatexfile(destinationfilename='latexfile0.tex',samplelabel='samplelabel',sampleparams='parameters',rfilelist=[],thetafilelist=[],twomhzfilelist=[],rpumpblockedfilelist=[],thetapumpblockedfilelist=[]):
	"""
	creates a .tex file which will have the desired images files organized in an appealing way.
	:param string: destinationfilename: The name of the .tex file to be created.
	:param string: samplelabel: Header at the top of the page
	:param string: sampleparams: Experimental params to be written below the header
	:param list string:rfilelist: list of strings with the file names of the amplitude images
	:param string::thetafilelist: list of strings with the file names of the phase images NOTE! thetafile list must be longer than rfilelist or this will crash!!!
	:param string::twomhzfilelist: list of strings with the file names of the 2MHz images

	"""
	with open(destinationfilename,'w+') as file:
		file.write('\\documentclass[12pt,a4paper]{article}\n')
		file.write('\\usepackage[margin=0.3in,bottom=0.1in]{geometry}\n')
		file.write('\\usepackage{graphicx}\n')
		file.write('\\usepackage{caption}\n')
		file.write('\\usepackage{multicol}\n')
		file.write('\\begin{document}\n')
		file.write('\\begin{multicols}{2}\n')
		file.write('[\n')
		file.write('\\section*{'+samplelabel+'}\n')
		file.write(sampleparams+'\n')
		if(twomhzfilelist != []):
			file.write(']\n')
			file.write('\\noindent 2 MHz\\\\\n')
			file.write('\\noindent R\\\\\n')
		for i in range(len(twomhzfilelist)):
			file.write('\\noindent\\includegraphics[width=9cm]{'+twomhzfilelist[i]+'}\\\\\n')
		if(rpumpblockedfilelist != []):
			file.write('\\noindent 1 MHz Pump Blocked\\\\\n')
			file.write('\\noindent R\\quad\\quad\\quad\\quad\\quad\\quad\\quad$\\quad$\\quad$\\theta$\\\\\n')
			for i in range(len(rpumpblockedfilelist)):
				file.write('\\noindent\\includegraphics[width=4.5cm]{'+rpumpblockedfilelist[i]+'}\n')
				file.write('\\includegraphics[width=4.5cm]{'+thetapumpblockedfilelist[i]+'}\\\\\n')
		file.write('\\noindent 1 MHz\\\\\n')
		file.write('\\noindent R\\quad\\quad\\quad\\quad\\quad\\quad\\quad$\\quad$\\quad$\\theta$\\\\\n')
		for i in range(len(rfilelist)):
			file.write('\\noindent\\includegraphics[width=4.5cm]{'+rfilelist[i]+'}\n')
			file.write('\\includegraphics[width=4.5cm]{'+thetafilelist[i]+'}\\\\\n')
			if(np.mod(i,5) is 0 and i>0):
				file.write('\\columnbreak\n')

		file.write('\\end{multicols}\n')
		file.write('\\end{document}\n')
		file.close()
		#Maybe wait for file to appear.
		if 1:
			time.sleep(3.6)
		return

def runlatexfile(filename='latexfile0.tex'):
	"""
	writes a command to the cmd line which runs the .tex file
	NOTE: this requires latex to run pdflatex.
	check to see if you have pdflatex:
	enter 'which pdflatex' into cmd line to get
	/Library/TeX/texbin/pdflatex
	get Mactex by following the instructions here
	https://youtu.be/MeeLtx_VcuI
	For debug on pdflatex see here
	https://superuser.com/questions/1038612/where-do-i-get-the-pdflatex-program-for-mac
	Homebrew may be helpful
	https://brew.sh/

	:param string: filename: The name of the .tex file to be run.
	"""
	cmd='pdflatex '+filename
	os.system(cmd)
	return

def writeexceldata(filename='data.xlsx',brx=[],bry=[],srx=[],sry=[],flu=[],timepoints=[],signal=[]):
	df = pd.DataFrame(data=np.transpose(signal),index=timepoints,columns=[srx,sry,brx,bry,flu*1000.])
	df.index.name = 'timepoints (ps)'
	df.to_excel(filename)
	return

"""
========================================================================================================
SECTION 2 For batch data loading and processing
========================================================================================================
Example Code

import pysuem as ps
import numpy as np
dp=ps.loadtimepoints("delaypoints20200204121823.txt")
tp=ps.formatdecimalstring(np.flip(ps.loadtimepoints("timepoints20200204121823.txt")),0)
tp0=np.flip(ps.loadtimepoints("timepoints20200204121823.txt"))
searchstring0="mypiranahnewc_ac_s-459_pump0.2mW_l2_162_fd2_tc30us100us_116_mag1000x"
destinationfileprefix="mypiranahnewbac_200uW"
channels0=['1','2']
power0=0.2
av0=15
mag0=1000
wd0=14
lambda0=532
polarizer0=160
reprate0=2
fdiv0=2
samplelabel0='SiO\\textsubscript{2} P-Type Wet Thermal Oxide Collected With Current Amplifier'
if 0:
	fldata=ps.getfilenames(searchstring0,index=dp,suffix='.txt',verbose=1)
	ps.processlongdatamulti(fldata,channels=channels0,verbose=1)
if 0:
	fldata_avg_ch2=ps.getfilenames(searchstring0,index=dp,suffix='_avg_ch2.txt',verbose=1)
	fldata_avg_ch1=ps.getfilenames(searchstring0,index=dp,suffix='_avg_ch1.txt',verbose=1)
	#xytmat=ps.assembletyxmat(fldata_avgch2,searchstring0+"ch"+channels0[0]+"_xytmat.txt")

	#get scale factors for ch1 and ch2
	sf_avg_ch2=ps.getstretchfactormulti(fldata_avg_ch2,minimalstretch=1)
	sf_avg_ch1=ps.getstretchfactormulti(fldata_avg_ch1,minimalstretch=1)
	#stretch ch1 images and ch2 images by the same values
	ps.stretchdatamulti(fldata_avg_ch2,sf_avg_ch2)
	ps.stretchdatamulti(fldata_avg_ch1,sf_avg_ch1)
if 1:
	#get the file names of the stretched images
	fldata_stretched_ch2=ps.getfilenames(searchstring0,index=dp,suffix='_avg_ch2_stretched.txt',verbose=1)
	fldata_stretched_ch1=ps.getfilenames(searchstring0,index=dp,suffix='_avg_ch1_stretched.txt',verbose=1)
	

	#note you have to index the first element of the list or the last elemnt of te list depending on ordering of delay points
	roich2=ps.clickimagecoords(fldata_stretched_ch2[0])
	maskch2=ps.createmask(roich2,show=1,savemask=1,destination=destinationfileprefix+"_ch2_mask.txt")
	ktch2=ps.averageROImulti(fldata_stretched_ch2,maskch2)
	#note we need to use the float version of timepoints rather than string version
	ps.plotkinetictrace(tp0,ktch2)
	ps.savetimeseries(tp0,ktch2,destination=destinationfileprefix+"ch2_ts.txt")
if 1:
	ps.createimagemulti(fldata_stretched_ch1)
	ps.createimagemulti(fldata_stretched_ch2)
	flim_stretched_ch2_tif=ps.getfilenames(searchstring0,index=dp,suffix='_avg_ch2_stretched.tif',verbose=1)
	flim_stretched_ch1_tif=ps.getfilenames(searchstring0,index=dp,suffix='_avg_ch1_stretched.tif',verbose=1)

	ps.addcaptionmulti(flim_stretched_ch2_tif,searchstring0+"_ch2",tp,prefix="_t",timepoints=tp,power=power0,wd=wd0,mag=mag0,av=av0)
	ps.addcaptionmulti(flim_stretched_ch1_tif,searchstring0+"_ch1",tp,prefix="_t",timepoints=tp,power=power0,wd=wd0,mag=mag0,av=av0)
	flim_ch2_cap=ps.getfilenames(searchstring0+"_ch2",index=tp,delimiter="_t",suffix='.tif',verbose=1)
	flim_ch1_cap=ps.getfilenames(searchstring0+"_ch1",index=tp,delimiter="_t",suffix='.tif',verbose=1)
	ps.converttojpeg(filelist=flim_ch2_cap,verbose=1)
	ps.converttojpeg(filelist=flim_ch1_cap,verbose=1)
if 1:
	flim_ch2_cap_jpeg=ps.getfilenames("mypiranahnewc_ac_s-459_pump200uW_l2_162_fd2_tc30us100us_116_mag1000x"+"_ch2",index=tp,delimiter="_t",suffix='.jpeg',verbose=1)
	flim_ch1_cap_jpeg=ps.getfilenames("mypiranahnewc_ac_s-459_pump200uW_l2_162_fd2_tc30us100us_116_mag1000x"+"_ch1",index=tp,delimiter="_t",suffix='.jpeg',verbose=1)
	ps.createlatexfile(destinationfilename=destinationfileprefix+'_Timeseries.tex',samplelabel=samplelabel0,sampleparams=str(av0)+' kV, Mag '+str(mag0)+'x, '+str(power0)+' mW, '+str(lambda0)+' nm, Polarizer '+str(polarizer0)+'\\textsuperscript{o}, Working Distance +'+str(wd0)+' mm, '+str(reprate0)+' MHz, Frequency Dividing '+str(fdiv0),rfilelist=flim_ch1_cap_jpeg,thetafilelist=flim_ch2_cap_jpeg)
	ps.runlatexfile(filename=destinationfileprefix+'_Timeseries.tex')

"""
def processlongdatamulti(filelist,channels=['1'],verbose=0,saveavgimagedata=1):
	"""
	A module for proscesssing a time series data set of SUEM images into a minimal datamatrix usually 738 by 485. 
	:param string filelist: list of names of data from which the destination files will be written with a suffix like _avg appended to them.
	:param: list string: channels: ['1','3'] which channels do you want to process. typically 1 MHz R and Theta are '1' and '2' while 2 MHz R and Theta are '3' and '4'.
	:param: boolean: verbose: print out debugging values
	:param: boolean: saveavgimagedata: basically must be turned on. says to save the resultant datamats.
	"""
	for f in filelist:
		if(verbose):
			start_time = time.time()
		data= loadlongdata(f,verbose=verbose)
		hcp,vcp=getcutpoints(data,verbose=verbose)
		for ch in channels:
			getdatamat(f,ch,data,hcp,vcp,saveavgimagedata=saveavgimagedata,verbose=verbose)
		if(verbose):
			print("--- %s seconds to process the channels --- " % (time.time() - start_time))
			print("%s" % f)
	return

def loadlongdata(filename,verbose=0):
	data= pd.read_csv(filename, delimiter = "\t",skiprows=2,names=["hdef", "vdef", "ch1", "ch2","ch3","ch4"])
	if(verbose):
		statinfo = os.stat(filename)
		print("The file:" +filename+" is ",statinfo.st_size, "bytes")
	return data

def getcutpoints(data, savecutpoints=0,verbose=0):
	"""
	A module for determining the cut points of the SUEM data from the horizontal and vertical deflectors. Data should be loaded from
	:param dataframe from pandas: data: 
	:param boolean: savecutpoints: option to save the cut points to a file could be compared and used later
	:param boolean: verbose: print debugging values
	"""
	if(verbose):
		print("getting cutpoints")
	hdef=data.loc[:,'hdef'].to_numpy()
	vdef=data.loc[:,'vdef'].to_numpy()
	dvdef=np.diff(vdef)
	std_dvdef=np.std(dvdef)
	max_dvdef=np.amax(dvdef)
	vindex=np.where(dvdef>std_dvdef+(max_dvdef-std_dvdef)/3)[0]
	vcp=np.delete(vindex,np.where(np.diff(vindex)<10)[0])
	dhdef=np.diff(hdef)
	std_dhdef=np.std(dhdef)
	max_dhdef=np.amax(dhdef)
	hindex=np.where(dhdef>std_dhdef+(max_dhdef-std_dhdef)/3)[0]
	hlarge=dhdef[hindex]
	hcp=np.delete(hindex,np.where(np.diff(hindex)<10)[0])
	if(savecutpoints):
		fhcp=generatefilename(f,"hcp",verbose=verbose)
		fvcp=generatefilename(f,"vcp",verbose=verbose)
		np.savetxt(fhcp.hcp,delimiter='\t', newline='\n')
		np.savetxt(fvcp,vcp,delimiter='\t', newline='\n')
	return hcp, vcp

def getdatamat(filename,channel,data,hcp,vcp,saveavgimagedata=1,saverawimagedata=0,saveprocessedimagedata=0,verbose=0,xpixel=738,ypixel=485):
	"""
	A module for adding a suffix to a filename before the file type extension
	If no file '.' is found an extension is added by default.
	:param string filename: file name of data from which the destination files will be written with a suffix like _avg appended to them.
	:param list string: channel: '1','2','3','4' which column of the data are you currently processing
	:param dataframe: data: the data extracted from read csv of pandas to be processed
	:param np.array: hcp: horizontal cut points
	:param np.array: vcp: vertical cut points
	:param boolean: verbose: print debugging values
	:param int: xpixel: how many pixels is in each horizontal sweep
	:param int: ypixel: how many horizontal sweeps are in each vertical sweep

	"""
	numimages=len(vcp)-1
	min_dhcp=np.amin(np.diff(hcp))
	chx=data.loc[:,'ch'+channel].to_numpy()
	unevendata=[l for l in np.split(chx,hcp)[1:-1]]
	#Tell the use how many vertical sweeps there are
	if(verbose):
		print(filename)
		print("--------There are "+str(numimages)+ " vertical sweeps to average--------")
	#warning this takes a long time to line by line write a giant datamat
	if(saverawimagedata):
		np.set_printoptions(threshold=sys.maxsize)
		f_raw=generatefilename(filename,"_raw_ch"+channel)
		file=open(f_raw,'w+')
		for i in unevendata:
			file.write(str(i[0:min_dhcp]))
		file.close()
	#Aspect ration is 738x485
	datamat=np.array([], dtype=float).reshape(0,738)
	if(saveprocessedimagedata):
		f_prc=generatefilename(filename,"_prc_ch"+channel)
		file2=open(f_prc,'w+')
		for i in unevendata:
			line=np.mean(i[:-np.mod(len(i),xpixel)].reshape((-1, xpixel),order='F'), axis=0)
			file2.write(str(line))
		file2.close()
	for i in unevendata:
		line=np.mean(i[:-np.mod(len(i),xpixel)].reshape((-1, xpixel),order='F'), axis=0)
		datamat=np.vstack((datamat,line))
	if(np.mod(np.shape(datamat),ypixel)[0]>0 and np.mod(np.shape(datamat),ypixel)[0]<242):
		counter=0
		if(verbose):
			print("deleting a line")
		while np.mod(np.shape(datamat),ypixel)[0]>0 and counter<10:
			counter=counter+1
			datamat=np.delete(datamat,-1,axis=0)

	elif(np.mod(np.shape(datamat),ypixel)[0]>242):
		counter=0
		if(verbose):
			print("adding a line")
		while np.mod(np.shape(datamat),ypixel)[0]>242 and counter<10:
			counter=counter+1
			datamat=np.vstack((datamat,np.zeros(xpixel)))

	datamat_avg=np.mean(datamat.reshape((-1,ypixel,xpixel)),axis=0)
	im_avg=Image.fromarray(np.uint8(datamat_avg*255))
	if(saveavgimagedata):
		f_avg=generatefilename(filename,"_avg_ch"+channel,verbose=verbose)
		np.savetxt(f_avg,datamat_avg,delimiter='\t', newline='\n')
	return datamat_avg

def assembletyxmat(filelist,destination,savetyxmat=0):
	"""assembles data to the correct format time by pixel y by pixel x
	:param string filelist: list of names of data from which the destination files will be written with a suffix like _avg appended to them.
	:param string: destination: path where image will be saved
	:param boolean: savetyxmat: option to save this assembled 3d matrix in addition to returning it.
	"""

	#first create the tyx matrix
	tyxlist=[]
	for f in filelist:
		tyxlist.append(np.loadtxt(f))
	tyxmat=np.array(tyxlist)
	#next save the tyxmatrix to a .txt file
	if(savetyxmat):
		with open(destination, 'w+') as outfile:
			outfile.write('# (image,y-pixel,x-pixel): {0}\n'.format(tyxmat.shape))
			for data_slice in tyxmat:
				# Writing out a break to indicate different slices...
				outfile.write('# New image\n')
				np.savetxt(outfile, data_slice,delimiter='\t', newline='\n', fmt='%-7.2f')
	return tyxmat

def loadtyxmat(filename,ypixel=485):
	"""
	load the tyxmatrix. no the load text only works with 2d data so reshape the loaded data.
	:param string filename: file name of data which will be loaded
	:param int ypixel: what is the dimension of the images in the y dimension.
	"""
	data=np.loadtxt(filename)
	return data.reshape(data.shape[0]//ypixel,ypixel,ypixel)


"""
=======================================================================================================
SECTION 3 Plotting & Image Generationg
========================================================================================================
"""


def createimage(datamat,save=1,show=0,destination="image",filetype='tif',rotate=0):
	"""
	ps.createimage(ie0,save=1,show=1,destination="ie0_mask1")
	"""
	im=Image.fromarray(np.uint8(datamat*255))
	if rotate==90:
		im=np.transpose(im)
	if rotate==180:
		im=np.flip(im,axis=0)
	if rotate==270:
		im=np.flip(np.transpose(im),axis=0)
	if(save):
		destinationname=generatefilename(destination,extension=filetype)
		im.save(destinationname)
	return

def createimagemulti(filelist,**keyword_parameters):
	if ('suffix' in keyword_parameters):
		suffix=keyword_parameters['suffix']
	else:
		suffix=""
	if ('suffix' in keyword_parameters):
		rotate=keyword_parameters['rotate']
	else:
		rotate=0
	for f in filelist:
		datamat=np.loadtxt(f)
		destination=generatefilename(f,suffix=suffix,extension='tif')
		createimage(datamat,save=1,destination=destination,rotate=rotate)
	return



def plotkinetictracemulti(filelist,timepoints,filelist2=[],save=1,show=1,autoclose=0,logx=1,xlim=[],destination="multikinetictrace.tif",**keyword_parameters):
	"""
	module for saving a set of timepoints and amplitudes to txt file.
	:param list float: timepoints: time delays coorespnding to delay stage postion.
	:param list float: kinetictrace: values of a kinetics traces relating to each time point. usually gotten from averageROI multi
	:param string: destinatiion: file name of the destination np.savetxt()
	Example:
	fl_ts_nb_ch4=ps.getfilenames(searchstring0,index=np.arange(20),delimiter="_er",suffix="_nb.txt",verbose=1)
	ts=ps.plotkinetictracemulti(fl_ts_nb_ch4,tp0,logx=False)
	"""
	@ticker.FuncFormatter
	def major_formatter(x, pos):
		return "%.1f"%x
	if ('labels' in keyword_parameters):
		labels=keyword_parameters['labels']
	if ('xlabel' in keyword_parameters):
		xlabel=keyword_parameters['xlabel']
	else:
		xlabel='Time Delay (ps)'
	if ('ylabel' in keyword_parameters):
		ylabel=keyword_parameters['ylabel']
	else:
		ylabel='Secondary Electron Contrast (A.U.)'
	if ('figsize' in keyword_parameters):
		figsize=keyword_parameters['figsize']
		fig, ax = plt.subplots(figsize=(figsize[0], figsize[1]))
	else:
		fig, ax = plt.subplots()
	if ('ytickrot' in keyword_parameters):
		print("rotate y ticks")
		plt.yticks(rotation=90)
		plt.gca().yaxis.set_major_formatter(major_formatter)
	if ('xtickrot' in keyword_parameters):
		plt.xticks(rotation=90)
		plt.gca().xaxis.set_major_formatter(major_formatter)
	colors = [plt.cm.jet(i) for i in np.linspace(0, 1, len(filelist))]
	ax.set_prop_cycle('color', colors)
	timeseries=np.array([], dtype=float).reshape(0,len(timepoints))
	counter=0
	for f in filelist:
		if ('labels' in keyword_parameters):
			label0=labels[counter]
		else:
			label0=f[f.rfind("_er")+3:f.rfind(".")]
		kt=np.loadtxt(f)
		plt.plot(timepoints, kt,marker='o',label=label0,linestyle = 'None')
		if logx:
			plt.xscale('symlog')
		timeseries=np.vstack((timeseries,kt))
		counter+=1
	if(filelist2 is not []):
		for f2 in filelist2:
			kt=np.loadtxt(f2)
			plt.plot(timepoints, kt,'--',color='k',linestyle = 'None')
	if(len(xlim)==2):
		plt.xlim(xlim[0],xlim[1])
		plt.xticks([10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000]) 
	#plt.legend(loc='best')
	plt.xlabel(xlabel,fontname="Arial")
	plt.ylabel(ylabel,fontname="Arial")
	plt.legend(loc='upper right',fancybox=True, framealpha=0.25)
	if save:
		plt.savefig(destination)
	if show:
		if(autoclose):
			plt.show(block=False)
			plt.pause(2)
			plt.close()
		else:
			plt.show()
	timeseries=np.vstack((timepoints,timeseries))
	timeseries.T
	return timeseries

def plotkinetictracemany(filelist,timepoints,filelist2=[],save=1,show=0,autoclose=0,logx=1,xlim=[],verbose=0,**keyword_parameters):
	"""
	module for saving a set of timepoints and amplitudes to txt file.
	:param list float: timepoints: time delays coorespnding to delay stage postion.
	:param list float: kinetictrace: values of a kinetics traces relating to each time point. usually gotten from averageROI multi
	:param string: destinatiion: file name of the destination np.savetxt()
	Example:
	fl_ts_nb_ch4=ps.getfilenames(searchstring0,index=np.arange(20),delimiter="_er",suffix="_nb.txt",verbose=1)
	ts=ps.plotkinetictracemulti(fl_ts_nb_ch4,tp0,logx=False)
	"""
	@ticker.FuncFormatter
	def major_formatter(x, pos):
		return "%.1f"%x
	for i in range(len(filelist)):
		if ('labels' in keyword_parameters):
			labels=keyword_parameters['labels']
		if ('figsize' in keyword_parameters):
			figsize=keyword_parameters['figsize']
			fig, ax = plt.subplots(figsize=(figsize[0], figsize[1]))
		else:
			fig, ax = plt.subplots(figsize=(3.25,3.25))
		if ('tight' in keyword_parameters):
			plt.tight_layout()
		if ('axispad' in keyword_parameters):
			axispad=keyword_parameters['axispad']
			ax.tick_params(axis='x', pad=axispad)
			ax.tick_params(axis='y', pad=axispad)
		if ('ytickrot' in keyword_parameters):
			print("rotate y ticks")
			plt.yticks(rotation=90)
			plt.gca().yaxis.set_major_formatter(major_formatter)
		if ('xtickrot' in keyword_parameters):
			plt.xticks(rotation=90)
			plt.gca().xaxis.set_major_formatter(major_formatter)
		colors = [plt.cm.jet(i) for i in np.linspace(0, 1, len(filelist))]
		ax.set_prop_cycle('color', colors)
		timeseries=np.array([], dtype=float).reshape(0,len(timepoints))
		counter=0
		if ('labels' in keyword_parameters):
			label0=labels[i]
		else:
			label0=f[f.rfind("_er")+3:f.rfind(".")]
		kt=np.loadtxt(filelist[i])
		plt.plot(timepoints, kt,marker='o',color=colors[i],label=label0)
		if(filelist2 is not []):
			kt_fit=np.loadtxt(filelist2[i])
			plt.plot(timepoints, kt_fit,'--',color='k')
		if logx:
			plt.xscale('symlog')
		timeseries=np.vstack((timeseries,kt))
		counter+=1
		if(len(xlim)==2):
			plt.xlim(xlim[0],xlim[1])
			plt.xticks([10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000,7500,8000]) 
		#plt.legend(loc='best')
		plt.xlabel('Time Delay (ps)',fontname="Arial",labelpad=-1.1)
		plt.ylabel('Secondary Electron Contrast (A.U.)',fontname="Arial",labelpad=-1.1)
		plt.legend()
		if save:
			f_plot=generatefilename(filelist[i],"_plot",extension="tif",verbose=verbose)
			plt.savefig(f_plot)
		if show:
			if(autoclose):
				plt.show(block=False)
				plt.pause(2)
				plt.close()
			else:
				plt.show()
	return




def plotROItracemulti(epallx,epally,filename=None,save=1,destination="ROIkinetictrace.tif",show=1,autoclose=0,**keyword_parameters):
	if ('figsize' in keyword_parameters):
		figsize=keyword_parameters['figsize']
		fig, ax = plt.subplots(figsize=(figsize[0], figsize[1]))
	else:
		fig, ax = plt.subplots()
	colors = [plt.cm.jet(i) for i in np.linspace(0, 1,np.shape(epallx)[0])]
	ax.set_prop_cycle('color', colors)
	plt.setp(ax.get_xticklabels(), visible=False)
	plt.setp(ax.get_yticklabels(), visible=False)
	ax.tick_params(axis='both', which='both', length=0)
	if ('labels' not in keyword_parameters):
		label=[str(k) for k in range(np.shape(epallx)[0])]
	else:
		label=keyword_parameters['labels']
	for i in range(np.shape(epallx)[0]):
		plt.plot(epallx[i,:], epally[i,:],label=label[i])
	if(filename is not None):
		datamat=np.loadtxt(filename)
		im=Image.fromarray(np.uint8(datamat*255))
		plt.imshow(im, cmap = 'gray')
	plt.legend(loc='best')
	if save:
		plt.savefig(destination)
	if show:
		if(autoclose):
			plt.show(block=False)
			plt.pause(2)
			plt.close()
		else:
			plt.show()
	return

def plotROItracemulti2(maskset,filename=None,save=1,destination="ROIkinetictrace.tif",show=1,autoclose=0,**keyword_parameters):
	#accepts a tuple of coordset rather than a list of x and a list of y
	m0=maskset[0]
	#print("SHAPE MASKSET[0])[1]",np.shape(maskset[0])[0])
	allx=ally=np.array([], dtype=float).reshape(0,np.shape(maskset[0])[0]+1)
	for m in maskset:
		#print("m[:,0]",np.append(m[:,0],m[0,0]))
		#print("m[:,1]",np.append(m[:,1],m[0,1]))
		allx=np.vstack((allx,np.append(m[:,0],m[0,0])))
		ally=np.vstack((ally,np.append(m[:,1],m[0,1])))
	if ('figsize' in keyword_parameters):
		figsize=keyword_parameters['figsize']
		fig, ax = plt.subplots(figsize=(figsize[0], figsize[1]))
	else:
		fig, ax = plt.subplots()
	colors = [plt.cm.jet(i) for i in np.linspace(0, 1,np.shape(allx)[0])]
	ax.set_prop_cycle('color', colors)
	plt.setp(ax.get_xticklabels(), visible=False)
	plt.setp(ax.get_yticklabels(), visible=False)
	ax.tick_params(axis='both', which='both', length=0)
	if ('labels' not in keyword_parameters):
		label=[str(k) for k in range(np.shape(allx)[0])]
	else:
		label=keyword_parameters['labels']
	for i in range(np.shape(allx)[0]):
		plt.plot(allx[i,:], ally[i,:],label=label[i])
	if(filename is not None):
		datamat=np.loadtxt(filename)
		im=Image.fromarray(np.uint8(datamat*255))
		plt.imshow(im, cmap = 'gray')
	plt.legend(loc='best',fancybox=True, framealpha=0.25)
	if save:
		plt.savefig(destination,bbox_inches='tight')
	if show:
		if(autoclose):
			plt.show(block=False)
			plt.pause(2)
			plt.close()
		else:
			plt.show()
	return


def plotkinetictrace(kinetictrace,timepoints,show=1,autoclose=0,logx=1,save=0,destination="timeseries.txt",**keyword_parameters):
	"""
	module for saving a set of timepoints and amplitudes to txt file.
	:param list float: timepoints: time delays coorespnding to delay stage postion.
	:param list float: kinetictrace: values of a kinetics traces relating to each time point. usually gotten from averageROI multi
	:param string: destinatiion: file name of the destination np.savetxt()
	Example:
	ps.plotkinetictrace(tp0,ie0_ts,save=1,destination="ie0_ts.txt")
	"""
	@ticker.FuncFormatter
	def major_formatter(x, pos):
		return "%.1f"%x
	if ('figsize' in keyword_parameters):
		figsize=keyword_parameters['figsize']
		fig, ax = plt.subplots(figsize=(figsize[0], figsize[1]))
	else:
		fig, ax = plt.subplots()
	if ('yerr' in keyword_parameters):
		yerr=keyword_parameters['yerr']
		plt.errorbar(timepoints,kinetictrace, color='r',yerr=yerr)
	else:
		if np.shape(kinetictrace)[0]>1:
			plt.plot(timepoints, kinetictrace, marker='o',color='r')
		else:
			plt.plot(timepoints, kinetictrace, marker='o',color='r')
	if logx:
		plt.xscale('symlog')
	if ('xlabel' in keyword_parameters):
		plt.xlabel(keyword_parameters['xlabel'])
	if ('ylabel' in keyword_parameters):
		plt.ylabel(keyword_parameters['ylabel'])
	if ('ylim' in keyword_parameters):
		ylim=keyword_parameters['ylim']
		plt.ylim(ylim[0],ylim[1])
	if ('xlim' in keyword_parameters):
		xlim=keyword_parameters['xlim']
		plt.xlim(xlim[0],xlim[1])
	if ('ytickrot' in keyword_parameters):
		print("rotate y ticks")
		plt.yticks(rotation=90)
		plt.gca().yaxis.set_major_formatter(major_formatter)
	if ('xtickrot' in keyword_parameters):
		plt.xticks(rotation=90)
		plt.gca().xaxis.set_major_formatter(major_formatter)
	timeseries=np.vstack((timepoints,kinetictrace)).T
	if save:
		fn_tif=getfileprefix(destination,delimiter=".",verbose=0)+".tif"
		fn_txt=getfileprefix(destination,delimiter=".",verbose=0)+".txt"
		plt.savefig(fn_tif)
		np.savetxt(fn_txt,timeseries,delimiter='\t', newline='\n',fmt='%.18e' ,header='time delay (ps)\t amplitude')
	if show:
		if(autoclose):
			plt.show(block=False)
			plt.pause(2)
			plt.close()
		else:
			plt.show()
	return timeseries


def plotlinecut(linecuts,waveoffset=0,save=0,destination="spatialProfile.tif",show=0,autoclose=1,**keyword_parameters):
	"""

	Example:
	flim_stretched_shifted_rot_ch2=ps.getfilenames(searchstring0,index=dp,suffix='_avg_ch2_stretched_shifted_rot.txt',verbose=1)
	hlc=ps.linecutmulti(flim_stretched_shifted_rot_ch2,lineregion=[10,110],show=0,save=1)
	shiftx,shifty=ps.getpositionaxis(mag0)
	p0=[-.2,4,0.3,0.09925566]
	bounds=([-.22,0.,.000,0.092],[.1,8,.7,.106])
	"""
	@ticker.FuncFormatter
	def major_formatter(x, pos):
		return "%.2f"%x
	if ('yerr' in keyword_parameters):
		yerr=keyword_parameters['yerr']
	if ('axis' not in keyword_parameters):
		if(linecuts.ndim>1):
			axis=np.arange(np.shape(linecuts)[1])
		else:
			axis=np.arange(len(linecuts))
	else:
		axis=keyword_parameters['axis']
	if ('label' not in keyword_parameters):
		label=[str(i) for i in range(np.shape(linecuts)[0])]
	else:
		label=keyword_parameters['label']
	if ('figdim' in keyword_parameters):
		fig, ax = plt.subplots(figsize=keyword_parameters['figdim'])
	else:
		fig, ax = plt.subplots()
	colors = [plt.cm.jet(i) for i in np.linspace(0, 1, np.shape(linecuts)[0])]
	ax.set_prop_cycle('color', colors)
	if(linecuts.ndim>1):
		for i in range(np.shape(linecuts)[0]):
			plt.plot(axis,linecuts[i,:]+waveoffset*i,label=label[i]+'')
		plt.legend(loc='best')
	else:
		plt.plot(axis,linecuts+waveoffset)
	if ('fitdata' in keyword_parameters):
		fitdata=keyword_parameters['fitdata']
		for i in range(np.shape(fitdata)[0]):
			if ('yerr' in keyword_parameters):
				plt.errorbar(axis,fitdata[i,:]+waveoffset*i,'k', yerr=yerr[i], uplims=True, lolims=True)
			else:
				plt.plot(axis,fitdata[i,:]+waveoffset*i,'k--')
	if ('title' in keyword_parameters):
		plt.title(keyword_parameters['title'])
	if ('xlabel' in keyword_parameters):
		plt.xlabel(keyword_parameters['xlabel'])
	if ('ylabel' in keyword_parameters):
		plt.ylabel(keyword_parameters['ylabel'])
	if ('ytickrot' in keyword_parameters):
		plt.yticks(rotation=90)
		plt.gca().yaxis.set_major_formatter(major_formatter)
	if ('xtickrot' in keyword_parameters):
		plt.xticks(rotation=90)
		plt.gca().xaxis.set_major_formatter(major_formatter)
	#plt.tight_layout()
	if save:
		plt.savefig(destination)
	if show:
		if(autoclose):
			plt.show(block=False)
			plt.pause(2)
			plt.close()
		else:
			plt.show()
	return

def plotairyfunction():
	x=np.linspace(-200,200,20001)
	a=1
	wx=50.4
	airy=4*(special.jv(1,((x+0.00000000000001)*2*2**.5/wx))/((x+0.00000000000001)*2*2**.5/wx))**2
	sinc=a*(np.sin(np.pi*((x+0.00000000000001)/wx/2**.5)))**2/(np.pi*((x+0.00000000000001)/wx/2**.5))**2
	gaussian=a*np.exp(-2*(x/wx)**2)
	plt.plot(x,sinc,label="Sinc^2 - sin(pi*(x/wx/2^.5)))^2/(pi*(x/wx/2^.5))^2")
	plt.plot(x,airy,label="Airy - 4*(jv(1,(x*2*2^.5/wx))/(x*2*2^.5/wx))^2")
	plt.plot(x,gaussian,label="Gaussian - exp(-2*(x/wx)^2)")
	plt.xlabel('Sample Position (um)',fontname="Arial")
	plt.ylabel('Intensty',fontname="Arial")
	plt.legend(loc='center left')
	plt.show()
	return

def plotftmat(ftmat,timegrid,flugrid,st=[],title="",save=1,show=0,autoclose=0,destination="ftmat.tif"):
	if st==[]:
		st=timegrid[0]
	ylabel="Fluence (x10^15 photons/cm^3)"
	xlabel="Time Delay (ps)"
	yticks=np.round(np.geomspace(flugrid[0],flugrid[-1],9),4)
	ytickloc=(np.log(yticks)-np.log(flugrid[0]))/(np.log(flugrid[-1])-np.log(flugrid[0]))
	xticks=np.linspace(timegrid[0],timegrid[-1],10)
	print(xticks)
	xtickloc=(xticks-st)/(timegrid[-1]-st)
	idt=(np.abs(timegrid - st)).argmin()
	fakey=np.linspace(0,1,len(flugrid))
	fakex=np.linspace(0,1,len(timegrid)-idt)
	fX,fY=np.meshgrid(fakex,fakey)
	X,Y=np.meshgrid(timegrid[idt:],flugrid)
	Z=ftmat.iloc[:,idt:]
	fig1, ax1 = plt.subplots()
	im = ax1.imshow(Z, interpolation='bilinear', cmap=cm.jet,origin='lower',extent=[0, 1,0, 1], vmin=0)
	plt.xticks(xtickloc,np.round(xticks,1),fontname="Arial")
	plt.yticks(ytickloc,yticks,fontname="Arial")
	plt.ylabel(ylabel,fontname="Arial")
	plt.xlabel(xlabel,fontname="Arial")
	cbar=fig1.colorbar(im, ax=ax1)
	cbar.ax.set_ylabel('Response Function (A.U.)',fontname="Arial")
	if(title is not ""):
		plt.title(title)
	if save:
		plt.savefig(destination)
	if show:
		if(autoclose):
			plt.show(block=False)
			plt.pause(2)
			plt.close()
		else:
			plt.show()
	return

def plotprofilemat(profilemat,shiftx,shifty,logz=0,title="",save=1,show=0,autoclose=0,destination="profilemat.tif"):
	#X,Y=np.meshgrid(shiftx,shifty)
	if logz:
		profilemat=np.log(profilemat)
	fig1, ax1 = plt.subplots()
	im = ax1.imshow(profilemat, interpolation='bilinear', cmap=cm.jet,origin='lower',extent=[shiftx[0], shiftx[-1],shifty[0], shifty[-1]],vmax=np.amax(profilemat), vmin=np.amin(profilemat))
	ylabel="Sample position (um)"
	xlabel="Sample position (um)"
	ax1.set_ylim(shifty[-1], shifty[0])
	plt.ylabel(ylabel,fontname="Arial")
	plt.xlabel(xlabel,fontname="Arial")
	cbar=fig1.colorbar(im, ax=ax1)
	cbar.ax.set_ylabel('mJ/cm^2',fontname="Arial")
	if(title is not ""):
		plt.title(title)
	if save:
		plt.savefig(destination)
	if show:
		if(autoclose):
			plt.show(block=False)
			plt.pause(2)
			plt.close()
		else:
			plt.show()
	return
"""
=======================================================================================================
SECTION 4 Subtracting,Stretching Rotating,Shifting,Phase Rotation
========================================================================================================
"""
#________________________________________________________Stretching________________________________________________________
def getstretchfactor(datamat,overstretch=1):
	"""
	Get stretch factors of a data matrix corresponding the the minimum and maximum of a matrix.

	"""
	r=np.amax(datamat)-np.amin(datamat)
	a=(np.amax(datamat)+np.amin(datamat))/2
	return [a-r/2./overstretch,a+r/2./overstretch]

def getstretchfactor2(datamat):
	"""
	Depricated module 4-6-2020
	Get stretch factors of a data matrix corresponding the the minimum and maximum of a matrix.
	"""
	return [np.amin(datamat),np.amax(datamat)]

def getstretchfactorfromclickmulti(filelist,referencefilename="", selectedimageindex=0,save=0,minimalstretch=0,maximalstretch=0,xpixel=738,ypixel=485, show=0):
	sf=np.array([], dtype=float).reshape(0,2)
	if(referencefilename is not ""):
		selectedfile=referencefilename
	else:
		selectedfile=filelist[selectedimageindex]
	print("selected file",selectedfile)
	print("click on the region of interest of minimum value")
	mincoords=clickimagecoords(selectedfile,numclicks=4,destination="stretchminroi.txt",saveclickcoords=save)
	print("click on the region of interest of maximum value")
	maxcoords=clickimagecoords(selectedfile,numclicks=4,destination="stretchmaxroi.txt",saveclickcoords=save)
	minmask=createmask(mincoords,xpixel=xpixel,ypixel=ypixel, show=show, save=save,destination="minmask.txt")
	maxmask=createmask(maxcoords,xpixel=xpixel,ypixel=ypixel, show=show, save=save,destination="maxmask.txt")
	for f in filelist:
		sf=np.vstack((sf,[averageROI(f,minmask),averageROI(f,maxmask)]))
	if(minimalstretch):
		return np.array([np.min(sf[:,0]),np.max(sf[:,1])])
	if(maximalstretch):
		return np.array([np.max(sf[:,0]),np.min(sf[:,1])])
	return sf

def getstretchfactormulti(filelist,minimalstretch=0,overstretch=1):
	"""
	get a stretch factor to a series of images
	:param string filelist: list of names of data files.
	:param boolean: minimalstretch. Do we want to stretch all the images individuallys or take the minimal stretch of all the images and apply it.
	Example:
	searchstring0="vonwhalfongold_CA_pump26mW_l2_160_fd2_tc30us100us_116_mag12000x"
	fldata_avg_ch2=ps.getfilenames(searchstring0,index=['0.0'],suffix='_avg_ch2.txt',verbose=0)
	overstretchfactor=np.array([.01,.02,.05,.1,.2,.5,1,2,5,10,20])
	for o in overstretchfactor:
		osf=ps.getstretchfactormulti(fldata_avg_ch2,minimalstretch=1,overstretch=o)
		print("osf: ",osf)
		ps.stretchdatamulti(fldata_avg_ch2,osf,save=1,verbose=0,flooroverstretch=1)
		fldata_stretched_ch2=ps.getfilenames(searchstring0,index=['0.0'],suffix='_avg_ch2_stretched.txt',verbose=0)
		ps.createimagemulti(fldata_stretched_ch2,suffix="_os"+str(o))
	"""
	#array of min max for each image
	sf=np.array([], dtype=float).reshape(0,2)
	for f in filelist:
		datamat=np.loadtxt(f)
		sf=np.vstack((sf,getstretchfactor(datamat,overstretch=overstretch)))
	if(minimalstretch):
		return np.array([np.min(sf[:,0]),np.max(sf[:,1])])
	return sf

def stretchdata(datamat,stretchfactor):
	return np.divide(np.subtract(datamat,stretchfactor[0]),(stretchfactor[1]-stretchfactor[0]))

def stretchdatamulti(filelist,stretchfactor,save=1,verbose=0,flooroverstretch=1,suffix="_stretched"):
	if(np.shape(stretchfactor)==(2,)):
		sf=np.multiply(np.ones((len(filelist),2)),stretchfactor)
	else:
		sf=stretchfactor
	counter=0
	for f in filelist:
		datamat=np.loadtxt(f)
		if(verbose):
			print(f)
			print(sf[counter,:])
		datamat_stretched=stretchdata(datamat,sf[counter,:])
		if flooroverstretch:
			datamat_stretched=np.where(np.where(datamat_stretched<0,0,datamat_stretched)>1,1,np.where(datamat_stretched<0,0,datamat_stretched))
		if(save):
			f_stretched=generatefilename(f,suffix,verbose=verbose)
			np.savetxt(f_stretched,datamat_stretched,delimiter='\t', newline='\n')
		counter=counter+1
	return


def getscalefactormulti(filelist,backgroundindex=0,multiplier=1):
	"""
	get a scale factor to a series of images
	:param string filelist: list of names of data files.
	:param boolean: minimalstretch. Do we want to stretch all the images individuallys or take the minimal stretch of all the images and apply it.
	Example:
	searchstring0="vonwhalfongold_CA_pump26mW_l2_160_fd2_tc30us100us_116_mag12000x"
	fldata_avg_ch2=ps.getfilenames(searchstring0,index=['0.0'],suffix='_avg_ch2.txt',verbose=0)
	overstretchfactor=np.array([.01,.02,.05,.1,.2,.5,1,2,5,10,20])
	for o in overstretchfactor:
		osf=ps.getstretchfactormulti(fldata_avg_ch2,minimalstretch=1,overstretch=o)
		print("osf: ",osf)
		ps.stretchdatamulti(fldata_avg_ch2,osf,save=1,verbose=0,flooroverstretch=1)
		fldata_stretched_ch2=ps.getfilenames(searchstring0,index=['0.0'],suffix='_avg_ch2_stretched.txt',verbose=0)
		ps.createimagemulti(fldata_stretched_ch2,suffix="_os"+str(o))
	"""
	f_bkg=filelist[backgroundindex]
	bkgmat=np.loadtxt(f_bkg)
	m=np.mean(bkgmat)
	cf=np.array([], dtype=float).reshape(0,1)
	cf2=np.array([multiplier*np.mean(np.loadtxt(f)) for f in filelist])
	print(cf2)
	for f in filelist:
		datamat=np.loadtxt(f)
		cf=np.vstack((cf,multiplier*np.mean(datamat)))
	return cf

def adjustdatamulti(filelist,offset=0,scale=1,save=1,verbose=0):
	for f in filelist:
		datamat=np.loadtxt(f)
		if(verbose):
			print(f)
		datamat_adj=np.add(np.multiply(datamat,scale),offset)
		if(save):
			f_adj=generatefilename(f,"_adj",verbose=verbose)
			np.savetxt(f_adj,datamat_adj,delimiter='\t', newline='\n')
	return

def subtractdatamulti(filelist,filelistdelta,scalefactor=[],savedata=1,saveimage=0,suffix="_sub"):
	if isinstance(filelist, str):
		filelist=[filelist]
	if isinstance(filelistdelta, str):
		filelistdelta=[filelistdelta]
	if len(filelistdelta)==1:
		filelistdelta=[filelistdelta[0] for s in range(len(filelist))]
	if scalefactor==[]:
		scalefactor=np.ones(len(filelist))
	counter=0
	for f in filelist:
		datamatdelta=np.loadtxt(filelistdelta[counter])
		print("scalefactor",scalefactor[counter])
		print("amax",np.amax(datamatdelta))
		datamatdelta=datamatdelta*scalefactor[counter]
		print("amax",np.amax(datamatdelta))
		datamat=np.loadtxt(f)
		datamat_sub=np.subtract(datamat,datamatdelta)
		fn_sub=generatefilename(f,suffix=suffix)
		if savedata:
			np.savetxt(fn_sub,datamat_sub)
		if saveimage:
			createimage(datamat_sub,save=1,destination=fn_sub,filetype='tif')
		counter+=1
	return

def dividedatamulti(filelist,filelistdelta,scalefactor=[],savedata=1,saveimage=0,suffix="_div"):
	if isinstance(filelist, str):
		filelist=[filelist]
	if isinstance(filelistdelta, str):
		filelistdelta=[filelistdelta]
	if len(filelistdelta)==1:
		filelistdelta=[filelistdelta[0] for s in range(len(filelist))]
	counter=0
	for f in filelist:
		datamatdelta=np.loadtxt(filelistdelta[counter])
		datamat=np.loadtxt(f)
		datamat_sub=np.divide(datamat,datamatdelta)
		fn_sub=generatefilename(f,suffix=suffix)
		if savedata:
			np.savetxt(fn_sub,datamat_sub)
		if saveimage:
			createimage(datamat_sub,save=1,destination=fn_sub,filetype='tif')
		counter+=1
	return

def clickimagecoords(filename,numclicks=4,destination="clickcoords.txt",save=0):
	"""
	A module for extracting coordinates from a series of clicks.
	param: string: filename: name of the image file to be displayed
	param: int: numclicks: number of coordinates to be extracted.

	filename,numcoords=4
	Example
	**************************************
	c0=clickimagecoords('plumehigh0.tif',4)
	"""
	def onclick(event):
		print('you clicked the coordinates:', event.xdata, event.ydata)
		X_coordinate = event.xdata
		Y_coordinate = event.ydata
		mutable_object['click'] = np.array([int(X_coordinate),int(Y_coordinate)])
		plt.close()
		return

	print("Click %s times going counter clockwise."% (numclicks))
	coords=np.array([], dtype=np.int64).reshape(0,2)
	splitname=filename.split(".")
	if(splitname[-1]=="tif" or splitname[-1]=="jpg" or splitname[-1]=="jpeg" or splitname[-1]=="pdf" or splitname[-1]=="bmp"):
		im = Image.open(filename)
	else:
		datamat=np.loadtxt(filename)
		im=Image.fromarray(np.uint8(datamat*255))
	for i in range(numclicks):	
		fig=plt.figure()
		plt.imshow(im,cmap='gray', vmin=0, vmax=255)
		plt.plot(coords[:,0],coords[:,1],color='r')
		plt.tight_layout()
		mutable_object = {}
		cid = fig.canvas.mpl_connect('button_press_event', onclick)
		plt.show()
		coords=np.vstack((coords,mutable_object['click']))
		plt.close()
	if(save):
		np.savetxt(destination,coords,delimiter='\t', newline='\n')
	plt.close()
	return coords

def clickimagecoordsmulti(filename,numregions=2,numclicks=4,destinationprefix="clickcoords",save=0):
	coords1= coords2=coords3=coords4=coords5=coords6=coords7=coords8=coords9=coords10=np.array([], dtype=np.int64).reshape(0,2)
	coords1=clickimagecoords(filename,numclicks=numclicks,destination=destinationprefix+"_roi0_coordset.txt",save=save)
	if(numregions==1):
		return coords1
	coords2=clickimagecoords(filename,numclicks=numclicks,destination=destinationprefix+"_roi1_coordset.txt",save=save)
	if(numregions==2):
		return coords1,coords2
	coords3=clickimagecoords(filename,numclicks=numclicks,destination=destinationprefix+"_roi2_coordset.txt",save=save)
	if(numregions==2):
		return coords1,coords2,coords3	
	coords4=clickimagecoords(filename,numclicks=numclicks,destination=destinationprefix+"_roi3_coordset.txt",save=save)
	if(numregions==4):
		return coords1,coords2,coords3,coords4
	coords5=clickimagecoords(filename,numclicks=numclicks,destination=destinationprefix+"_roi4_coordset.txt",save=save)
	if(numregions==5):
		return coords1,coords2,coords3,coords4,coords5
	coords6=clickimagecoords(filename,numclicks=numclicks,destination=destinationprefix+"_roi5_coordset.txt",save=save)
	if(numregions==6):
		return coords1,coords2,coords3,coords4,coords5,coords6
	coords7=clickimagecoords(filename,numclicks=numclicks,destination=destinationprefix+"_roi6_coordset.txt",save=save)
	if(numregions==7):
		return coords1,coords2,coords3,coords4,coords5,coords6,coords7
	coords8=clickimagecoords(filename,numclicks=numclicks,destination=destinationprefix+"_roi7_coordset.txt",save=save)
	if(numregions==8):
		return coords1,coords2,coords3,coords4,coords5,coords6,coords7,coords8
	coords9=clickimagecoords(filename,numclicks=numclicks,destination=destinationprefix+"_roi8_coordset.txt",save=save)
	if(numregions==9):
		return coords1,coords2,coords3,coords4,coords5,coords6,coords7,coords8,coords9
	coords10=clickimagecoords(filename,numclicks=numclicks,destination=destinationprefix+"_roi9_coordset.txt",save=save)
	if(numregions==10):
		return coords1,coords2,coords3,coords4,coords5,coords6,coords7,coords8,coords9,coords10
	return


def getcoordsfromclickmulti(filelist,save=0,destination='clickcoords.txt'):
	"""
	"""
	sv0=np.array([], dtype=float).reshape(0,2)
	for f in filelist:
		datamat=np.loadtxt(f)
		sv0=np.vstack((sv0,np.flip(clickimagecoords(f,numclicks=1))))
	if save:
		np.savetxt(destination,sv0)
	return sv0

#________________________________________________________Phase rotation ________________________________________________________

def rotatephasemat(thetamat,deltatheta):
	return np.mod(np.add(thetamat,deltatheta),2*np.pi)

def voltage2radians(thetamat,minvoltage=-10,maxvoltage=10):
	return np.multiply(thetamat,2*np.pi/(maxvoltage-minvoltage))

def radians2voltage(thetamat,minvoltage=-10,maxvoltage=10):
	return np.add(np.multiply(thetamat,(maxvoltage-minvoltage)/(2*np.pi)),minvoltage)

def rtheta2x(rmat,thetamat):
	return np.multiply(rmat,np.cos(thetamat))

def rtheta2y(rmat,thetamat):
	return np.multiply(rmat,np.sin(thetamat))

def findphaseofminstd(rmat,thetamat,intervals=16,iterations=4,minvoltage=-10,maxvoltage=10,maxy=1,verbose=0):
	roi=np.array([0,2*np.pi])
	tm_rad=voltage2radians(thetamat,minvoltage=minvoltage,maxvoltage=maxvoltage)
	for i in range(iterations):
		thetagrid=np.linspace(roi[0],roi[1]-roi[1]/intervals,intervals)
		if maxy:
			temp_std=np.array([np.std(rtheta2y(rmat,rotatephasemat(tm_rad,t))) for t in thetagrid])
			mindeltatheta_index=temp_std.argsort()[-1]
		else:
			temp_std=np.array([np.std(rtheta2x(rmat,rotatephasemat(tm_rad,t))) for t in thetagrid])
			mindeltatheta_index=temp_std.argsort()[0]
		roi=np.array([thetagrid[np.mod(mindeltatheta_index-1,intervals)],thetagrid[np.mod(mindeltatheta_index+1,intervals)]])
		if verbose:
			print("iteration: ",i)
			print("theta values: ",thetagrid)
			print("std values: ",temp_std)
			print("opt theta", thetagrid[mindeltatheta_index])
			print("opt std", temp_std[mindeltatheta_index])
	return thetagrid[mindeltatheta_index]

def getphasefactormulti(rfilelist,thetafilelist,index=-1,useindex=1,usemin=0,intervals=8,iterations=4,minvoltage=-10,maxvoltage=10,maxy=1,verbose=0):
	if useindex:
		rmat0=np.loadtxt(rfilelist[index])
		thetamat0=np.loadtxt(thetafilelist[index])
		deltatheta0=findphaseofminstd(rmat0,thetamat0,intervals=intervals,iterations=iterations,minvoltage=minvoltage,maxvoltage=maxvoltage,verbose=verbose)
		phasefactor=np.ones(len(rfilelist))*deltatheta0
	else:
		phasefactor=array([], dtype=float64)
		for i in range(len(rfilelist)):
			rmat0=np.loadtxt(rfilelist[i])
			thetamat0=np.loadtxt(thetafilelist[i])
			phasefactor=np.append(phasefactor,findphaseofminstd(rmat0,thetamat0,intervals=intervals,iterations=iterations,minvoltage=minvoltage,maxvoltage=maxvoltage,maxy=maxy,verbose=verbose) , axis=0)
		if usemin:
			minphasefactor=np.amin(phasefactor)
			phasefactor=np.ones(len(rfilelist))*minphasefactor
	return phasefactor

def getphasefactorfromclick(rfilelist,thetafilelist,index=-1,minvoltage=-10,maxvoltage=10,showy=1,save=0,destination='phasefactors.txt',verbose=0):
	def onclick_select(event):
		y=np.round((float(str(event.inaxes).split("(")[1].split(";")[0].split(",")[1])-0.119882)/(0.320751-0.119882))
		x=np.round(((float(str(event.inaxes).split("(")[1].split(";")[0].split(",")[0])-.125)/(0.327174-0.125)))
		mutable_object['click'] = int(4.0*(3.0-y)+x)
		plt.close()
		return
	intervals=16
	roi=np.array([0,2*np.pi])
	thetagrid=np.linspace(roi[0],roi[1]-[1]/intervals,intervals)
	rmat=np.loadtxt(rfilelist[index])
	thetamat=np.loadtxt(thetafilelist[index])
	tm_rad=voltage2radians(thetamat,minvoltage=minvoltage,maxvoltage=maxvoltage)
	#fig=plt.figure(figsize=(11.5,7))
	fig=plt.figure()
	columns = 4
	rows = intervals/4
	for i in range(intervals):
		ax  = fig.add_subplot(rows,columns,i+1)
		ymat=rtheta2y(rmat,rotatephasemat(tm_rad,thetagrid[i]))
		sfy=getstretchfactor(ymat)
		ymat_streched=stretchdata(ymat,sfy)
		im=Image.fromarray(np.uint8(ymat_streched*255))
		plt.imshow(im, cmap = 'gray')
		plt.title(str(formatdecimalstring(thetagrid[i]*180/np.pi,1))), plt.xticks([]), plt.yticks([])
	fig.canvas.mpl_connect("button_press_event",onclick_select)
	mutable_object = {}
	plt.show()
	subplotindex = mutable_object['click']
	if verbose:
		print("subplot index: ",subplotindex)
		print("selected phase: ",thetagrid[subplotindex]*180/np.pi," degrees")
	phasefactor=np.ones(len(rfilelist))*thetagrid[subplotindex]
	if save:
		np.savetxt(destination,phasefactor)
	return phasefactor

def flipphasefactor(phasefactor):
	return rotatephasemat(phasefactor,np.pi) 

def rotatephasenmulti(rfilelist,thetafilelist,phasefactor,flipphase=0,minvoltage=-10,maxvoltage=10,save=1,verbose=0):
	if flipphase:
		rotatephasemat(phasefactor,np.pi)
	for i in range(len(rfilelist)):
		rmat0=np.loadtxt(rfilelist[i])
		thetamat0=np.loadtxt(thetafilelist[i])
		tm_rad=voltage2radians(thetamat0,minvoltage=minvoltage,maxvoltage=maxvoltage)
		tm_rot=rotatephasemat(tm_rad,phasefactor[i])
		y_mat0=rtheta2y(rmat0,tm_rot)
		x_mat0=rtheta2x(rmat0,tm_rot)
		if(save):
			f_phasedy=generatefilename(thetafilelist[i],"_phasedy",verbose=verbose)
			np.savetxt(f_phasedy,y_mat0,delimiter='\t', newline='\n')
			f_phasedx=generatefilename(thetafilelist[i],"_phasedx",verbose=verbose)
			np.savetxt(f_phasedx,x_mat0,delimiter='\t', newline='\n')
	return

#________________________________________________________Rotating & Shifting________________________________________________________
def rotatedatamulti(filelist,theta,radians=0,save=1,verbose=0,suffix="_rot",order=5):
	if radians:
		theta=theta*np.pi/180.
	for f in filelist:
		datamat=np.loadtxt(f)
		cval0=np.average(datamat)
		if(verbose):
			print(f)
			print(sf[counter,:])
		datamat_rot=ndimage.rotate(datamat, theta,reshape=False,mode='constant', cval=cval0)
		if(save):
			f_rot=generatefilename(f,suffix,verbose=verbose)
			np.savetxt(f_rot,datamat_rot,delimiter='\t', newline='\n')
	return

def getshiftvector(im0,im1):
	"""
	https://en.wikipedia.org/wiki/Phase_correlation
	https://stackoverflow.com/questions/2831527/phase-correlation/12253475
	https://www.lfd.uci.edu/~gohlke/code/imreg.py.html
	"""
	"""Return translation vector to register images."""
	shape = im0.shape
	f0 = fft2(im0)
	f1 = fft2(im1)
	ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
	t0, t1 = np.unravel_index(np.argmax(ir), shape)
	if t0 > shape[0] // 2:
		t0 -= shape[0]
	if t1 > shape[1] // 2:
		t1 -= shape[1]
	return [-t0,-t1]

def getshiftvectorfromclickmulti(filelist,save=0,shifttocenter=0,useindex=0,destination='stretchvector.txt'):
	"""
	Example code
	fldata_avg_stretched_ch3=ps.getfilenames(searchstring0,index=dp,suffix='_avg_ch3_stretched.txt',verbose=1)
	svch3_stretched2=ps.getshiftvectorfromclickmulti(fldata_avg_stretched_ch3)
	print(svch3_stretched2)
	ps.shiftdatamulti(fldata_avg_stretched_ch3, svch3_stretched2,verbose=1)
	flim_stretched_shifted_ch3=ps.getfilenames(searchstring0,index=dp,suffix='_avg_ch3_stretched_shifted.txt',verbose=1)
	ps.createimagemulti(flim_stretched_shifted_ch3)
	flim_stretched_shifted_ch3_tif=ps.getfilenames(searchstring0,index=dp,suffix='_avg_ch3_stretched_shifted.tif',verbose=1)
	ps.generategif(flim_stretched_shifted_ch3_tif,destinationfileprefix+'_stretched_shifted_clicks.gif',duration=150)
	"""
	
	if useindex>0:
		f=filelist[useindex]
		sv0= np.multiply(np.ones((len(filelist),2)),np.flip(clickimagecoords(f,numclicks=1)))
	else:
		sv0=np.array([], dtype=float).reshape(0,2)
		for f in filelist:
			datamat=np.loadtxt(f)
			sv0=np.vstack((sv0,np.flip(clickimagecoords(f,numclicks=1))))
	if shifttocenter:
		print(np.divide(np.shape(datamat),2.))
		sv1=np.round(np.subtract(sv0,np.round(np.divide(np.shape(datamat),2.)))).astype(int)
	else:
		sv1=np.round(np.subtract(sv0,sv0[0,:])).astype(int)
	if save:
		np.savetxt(destination,sv1)
		print("sv1:", sv1)
	return sv1


def getshiftvectormulti(filelist,useaverage=0):
	"""
	Example Code
	fldata_avg_stretched_ch3=ps.getfilenames(searchstring0,index=dp,suffix='_avg_ch3_stretched.txt',verbose=1)
	#svch3_stretched=ps.getshiftvectormulti(fldata_avg_stretched_ch3,useaverage=0)
	#ps.shiftdatamulti(fldata_avg_stretched_ch3, svch3_stretched,verbose=1)
	#flim_stretched_shifted_ch3=ps.getfilenames(searchstring0,index=dp,suffix='_avg_ch3_stretched_shifted.txt',verbose=1)
	#ps.createimagemulti(flim_stretched_shifted_ch3)
	#flim_stretched_shifted_ch3_tif=ps.getfilenames(searchstring0,index=dp,suffix='_avg_ch3_stretched_shifted.tif',verbose=1)
	#ps.generategif(flim_stretched_shifted_ch3_tif,destinationfileprefix+'_stretched_shifted1.gif',duration=150)
	"""
	sv=np.array([], dtype=float).reshape(0,2)
	if useaverage:
		print("Shifting images to match mean of filelist")
		tyxmat=assembletyxmat(filelist,savetyxmat=0)
		original=np.mean(tyxmat,axis=2)
	else:
		print("Shifting images to match:", filelist[0])
		original=np.loadtxt(filelist[0])
	for f in filelist:
		datamat=np.loadtxt(f)
		sv=np.vstack((sv,getshiftvector(original,datamat)))
	return sv

def getplumeshiftvector(datamat):
	x=np.arange(np.shape(datamat)[1])
	y=np.arange(np.shape(datamat)[0])
	midx=np.shape(datamat)[1]/2
	midy=np.shape(datamat)[0]/2
	X,Y=np.meshgrid(x,y)
	d0=np.sum(datamat)+.00000000001
	bXb=np.sum(np.multiply(datamat,X))/d0
	bYb=np.sum(np.multiply(datamat,Y))/d0
	return [np.round(bYb-midy,0),np.round(bXb-midx,0)]

def calculatecontrastflat(datamata,datamatb):
	Ia=np.sum(datamata)
	Ib=np.sum(datamatb)
	return 2*(Ia-Ib)/(Ia+Ib)

def calculatecontrastmask(datamata,datamatb,mask):
	Ia=np.sum(np.multiply(datamata,mask))
	Ib=np.sum(np.multiply(datamatb,mask))
	return 2*(Ia-Ib)/(Ia+Ib)

def calculatecontrastmaskmulti(filelista,filelistb,mask):
	for fa in filelista:
		for fb in filelistb:
			datamata=np.loadtxt(fa)
			datamata=np.loadtxt(fb)
	return

def showplume(plume,vmin=0.0, vmax=1.0,flip=1):
	if flip:
		plume = np.flip(np.transpose(plume.reshape(256,256)),axis=0)
	vmin=np.amin(plume)
	vmax=np.amax(plume)
	plt.imshow(plume, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
	plt.axis('off')

def applycontrastmask(datamat,mask,scale=0,offset=0):
	if scale:
		mask=mask/np.sum(np.abs(mask))
	if offset:
		mask=mask-np.mean(mask)
	return np.sum(np.multiply(datamat,mask))

def applycontrastmaskmulti(filelist,mask,scale=0,offset=0):
	mi=np.array([], dtype=float).reshape(0,1)
	for f in filelist:
		datamat=np.loadtxt(f)
		mi=np.vstack((mi,applycontrastmask(datamat,mask,scale=scale,offset=offset)))
	return mi

def applymulticomponentcontrastmask(filelist,masklist,save=0,verbose=0,scale=0,offset=0):
	mj=np.array([], dtype=float).reshape(len(filelist),0)
	for m in masklist:
		mk=np.loadtxt(m)
		mi=applycontrastmaskmulti(filelist,mk,scale=scale,offset=offset)
		print(np.shape(mi))
		print(np.shape(mj))
		mj=np.hstack((mj,mi))
		if save:
			fn_mi=generatefilename(m,suffix="_mi",verbose=verbose)
			np.savetxt(fn_mi,mi)
	if save:
			fn_mi_all=generatefilename(masklist[0],suffix="_mi_all",verbose=verbose)
			np.savetxt(fn_mi_all,mj)
	return mj

def getplumeshiftvectormulti(filelist,useindex=0):
	"""
	"""
	sv=np.array([], dtype=float).reshape(0,2) 
	for f in filelist:
		datamat=np.loadtxt(f)
		sv=np.vstack((sv,getplumeshiftvector(datamat)))
	if useindex!=0:
		sv2= np.multiply(np.ones((len(filelist),2)),sv[useindex,:])
		return sv2
	return sv

def cropdatamulti(fl,rx=256,ry=256,usemid=1,save=1,verbose=0,rotate=0):
	datamat0=np.loadtxt(fl[0])
	midx=np.shape(datamat0)[1]/2
	midy=np.shape(datamat0)[0]/2
	for f in fl:
		datamat=np.loadtxt(f)
		if usemid:
			datamat_cropped=datamat[int(midy-ry/2):int(midy+ry/2),int(midx-rx/2):int(midx+rx/2)]
		else:
			datamat_cropped=datamat[0:ry,0:rx]
		if rotate:
			datamat_cropped=np.flip(np.transpose(datamat_cropped))
		if save:
			f_cropped=generatefilename(f,"_cropped",verbose=verbose)
			np.savetxt(f_cropped,datamat_cropped)
	return

def shiftdata(datamat, shiftvector,fillwithaverage=1,verbose=0):
	"""
	datamat_shifted=shiftdata(datamat,sv[counter,:])
	"""
	dmsh=np.roll(np.roll(datamat, int(-shiftvector[0]),axis=0),int(-shiftvector[1]),axis=1)
	if(fillwithaverage):
		a=np.mean(datamat)
		#probably a nicer way to index the correct rows from 0 to positive or from negative to length
		if(shiftvector[1]>0):
			dmsh[:,-int(shiftvector[1]):]=a
		else:
			dmsh[:,:-int(shiftvector[1])]=a
		if(shiftvector[0]>0):
			dmsh[-int(shiftvector[0]):,:]=a
		else:
			dmsh[:-int(shiftvector[0]),:]=a
	return dmsh

def shiftdatamulti(filelist, shiftvector,saveshiftedimagedata=1,verbose=0):
	"""
	ps.shiftdatamulti(fldata_avg_stretched_ch3, svch3_stretched3,verbose=1)
	"""
	if(np.shape(shiftvector)==(2,)):
		sv=np.multiply(np.ones((len(filelist),2)),shiftvector)
	else:
		sv=shiftvector
	counter=0
	for f in filelist:
		datamat=np.loadtxt(f)
		if(verbose):
			print(f)
			print(sv[counter,:])
		datamat_shifted=shiftdata(datamat,sv[counter,:])
		if(saveshiftedimagedata):
			f_shifted=generatefilename(f,"_shifted",verbose=verbose)
			print(f_shifted)
			np.savetxt(f_shifted,datamat_shifted,delimiter='\t', newline='\n')
		counter=counter+1
	return


def similarity(im0, im1,applytranslation=1,applyrotation=0,applyscale=0):
	"""Return similarity transformed image im1 and transformation parameters.

	Transformation parameters are: isotropic scale factor, rotation angle (in
	degrees), and translation vector.

	A similarity transformation is an affine transformation with isotropic
	scale and without shear.

	Limitations:
	Scale change must be less than 1.8.
	No subpixel precision.

	"""
	if im0.shape != im1.shape:
		raise ValueError('images must have same shapes')
	if len(im0.shape) != 2:
		raise ValueError('images must be 2 dimensional')

	f0 = fftshift(abs(fft2(im0)))
	f1 = fftshift(abs(fft2(im1)))

	h = highpass(f0.shape)
	f0 *= h
	f1 *= h
	del h

	f0, log_base = logpolar(f0)
	f1, log_base = logpolar(f1)

	f0 = fft2(f0)
	f1 = fft2(f1)
	r0 = abs(f0) * abs(f1)
	ir = abs(ifft2((f0 * f1.conjugate()) / r0))
	i0, i1 = np.unravel_index(np.argmax(ir), ir.shape)
	angle = 180.0 * i0 / ir.shape[0]
	scale = log_base ** i1

	if scale > 1.8:
		ir = abs(ifft2((f1 * f0.conjugate()) / r0))
		i0, i1 = np.unravel_index(np.argmax(ir), ir.shape)
		angle = -180.0 * i0 / ir.shape[0]
		scale = 1.0 / (log_base ** i1)
		if scale > 1.8:
			raise ValueError('images are not compatible. Scale change > 1.8')

	if angle < -90.0:
		angle += 180.0
	elif angle > 90.0:
		angle -= 180.0
	if(applyscale):
		im2 = ndii.zoom(im1, 1.0/scale)
	else:
		im2=im1
	if(applyrotation):
		im2 = ndii.rotate(im2, angle)

	if im2.shape < im0.shape:
		t = np.zeros_like(im0)
		t[:im2.shape[0], :im2.shape[1]] = im2
		im2 = t
	elif im2.shape > im0.shape:
		im2 = im2[:im0.shape[0], :im0.shape[1]]


	f0 = fft2(im0)
	f1 = fft2(im2)
	ir = abs(ifft2((f0 * f1.conjugate()) / (abs(f0) * abs(f1))))
	t0, t1 = np.unravel_index(np.argmax(ir), ir.shape)

	if t0 > f0.shape[0] // 2:
		t0 -= f0.shape[0]
	if t1 > f0.shape[1] // 2:
		t1 -= f0.shape[1]
	if(applytranslation):
		im2 = ndii.shift(im2, [t0, t1])

	# correct parameters for ndimage's internal processing
	if angle > 0.0:
		d = int(int(im1.shape[1] / scale) * math.sin(math.radians(angle)))
		t0, t1 = t1, d+t0
	elif angle < 0.0:
		d = int(int(im1.shape[0] / scale) * math.sin(math.radians(angle)))
		t0, t1 = d+t1, d+t0
	scale = (im1.shape[1] - 1) / (int(im1.shape[1] / scale) - 1)

	return im2, scale, angle, [-t0, -t1]


def logpolar(image, angles=None, radii=None):
	"""Return log-polar transformed image and log base."""
	shape = image.shape
	center = shape[0] / 2, shape[1] / 2
	if angles is None:
		angles = shape[0]
	if radii is None:
		radii = shape[1]
	theta = np.empty((angles, radii), dtype='float64')
	theta.T[:] = np.linspace(0, np.pi, angles, endpoint=False) * -1.0
	# d = radii
	d = np.hypot(shape[0] - center[0], shape[1] - center[1])
	log_base = 10.0 ** (math.log10(d) / (radii))
	radius = np.empty_like(theta)
	radius[:] = np.power(log_base,
							np.arange(radii, dtype='float64')) - 1.0
	x = radius * np.sin(theta) + center[0]
	y = radius * np.cos(theta) + center[1]
	output = np.empty_like(x)
	ndii.map_coordinates(image, [x, y], output=output)
	return output, log_base
"""
=======================================================================================================
SECTION 5 Masking, Averaging, Regions of Interest
========================================================================================================
"""

#________________________________________________________Circles and ellipses ________________________________________________________
def createmask(coords,xpixel=738,ypixel=485, show=0, save=1,destination="mask.txt"):
	#https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
	p = path.Path(coords)  # square with legs length 1 and bottom left corner at the origin
	xv,yv = np.meshgrid(np.arange(xpixel),np.arange(ypixel))
	flags = p.contains_points(np.hstack((xv.flatten()[:,np.newaxis],yv.flatten()[:,np.newaxis])))
	mask=flags.reshape(ypixel,xpixel)
	if show:
		plt.imshow(mask,interpolation='nearest',cmap='binary')
		plt.show(block=False)
		plt.pause(2)
		plt.close()
	if save:
		np.savetxt(destination,mask,delimiter='\t', newline='\n')
	return mask

def createmaskmulti(coordset,xpixel=738,ypixel=485, show=0, save=1,destinationprefix="mask"):
	#coordset is a tuple
	for i in range(len(coordset)):
		coords=coordset[i]
		createmask(coords,xpixel=xpixel,ypixel=ypixel, show=show, save=save,destination=destinationprefix+str(i)+".txt")
	return

def createannularmask(center, big_radius, small_radius,xpixel=738,ypixel=485, show=0, save=0,destination="mask.txt"):
	Y, X = np.ogrid[:xpixel, :ypixel]
	distance_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
	mask = (small_radius <= distance_from_center) &(distance_from_center <= big_radius)
	if show:
		plt.imshow(mask,interpolation='nearest',cmap='binary')
		plt.show(block=False)
		plt.pause(2)
		plt.close()
	if save:
		np.savetxt(destination,mask,delimiter='\t', newline='\n')
	return mask

def applycircularmaskmulti(filelist,center,radius,xpixel=738,ypixel=485, show=0,save=0,destination="mask.txt",suffix="_ec.txt"):
	if isinstance(center,float):
		centers=np.multiply(np.ones(2,len(filelist)),center)
	else:
		centers=center
	print(centers)
	intervalvals=[]
	Y, X = np.ogrid[:ypixel, :xpixel]
	for i in range(len(filelist)):
		f=filelist[i]
		data=np.loadtxt(f)
		distance_from_center = np.sqrt((X - centers[i,1])**2 + (Y-centers[i,0])**2)
		tm = (distance_from_center <= radius)
		if show:
			plt.imshow(tm,interpolation='nearest',cmap='binary')
			plt.show(block=False)
			plt.pause(2)
			plt.close()
		intervalvals.append(averageROI(f,tm))
	if save:
		fn_pre=getfileprefix(filelist[0],delimiter="_")+suffix
		np.savetxt(fn_pre,intervalvals)
	return intervalvals

def createannularellipsemask(center, big_radiusx,big_radiusy, small_radiusx,small_radiusy,theta=0,xpixel=738,ypixel=485,radians=False,show=0, save=0,destination="mask.txt"):
	"""
	ie0=ps.createannularellipsemask(centercoord2, 84,49, .1,.1,theta=5)
	"""
	if(small_radiusx==0):
		small_radiusx=1
	if(big_radiusx==0):
		big_radiusx=1
	if(small_radiusy==0):
		small_radiusy=1
	if(big_radiusy==0):
		big_radiusy=1
	if(radians==False):
		theta=theta*3.1415259/180
	Y, X = np.ogrid[:ypixel, :xpixel]
	XmC=X-center[0]
	YmC=Y-center[1]
	distance_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
	mask = (1 <= (np.cos(theta)*(XmC)-np.sin(theta)*(YmC))**2/small_radiusx**2+(np.sin(theta)*(XmC)+np.cos(theta)*(YmC))**2/small_radiusy**2) &(1 >= (np.cos(theta)*(XmC)-np.sin(theta)*(YmC))**2/big_radiusx**2+ (np.sin(theta)*(XmC)+np.cos(theta)*(YmC))**2/big_radiusy**2)
	if show:
		plt.imshow(mask,interpolation='nearest',cmap='binary')
		plt.show(block=False)
		plt.pause(2)
		plt.close()
	if save:
		np.savetxt(destination,mask,delimiter='\t', newline='\n')
	return mask

def createannularhalfellipsemask(center, big_radiusx,big_radiusy, small_radiusx,small_radiusy,theta=0,xpixel=738,ypixel=485,radians=False,show=0, save=0,destination="mask.txt"):
	"""
	ie0=ps.createannularellipsemask(centercoord2, 84,49, .1,.1,theta=5)
	"""
	if(small_radiusx==0):
		small_radiusx=1
	if(big_radiusx==0):
		big_radiusx=1
	if(small_radiusy==0):
		small_radiusy=1
	if(big_radiusy==0):
		big_radiusy=1
	if(radians==False):
		theta=theta*3.1415259/180
	Y, X = np.ogrid[:ypixel, :xpixel]
	XmC=X-center[0]
	YmC=Y-center[1]
	distance_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
	mask_lower= np.multiply((1 <= (np.cos(theta)*(XmC)-np.sin(theta)*(YmC))**2/small_radiusx**2+(np.sin(theta)*(XmC)+np.cos(theta)*(YmC))**2/small_radiusy**2) &(1 >= (np.cos(theta)*(XmC)-np.sin(theta)*(YmC))**2/big_radiusx**2+ (np.sin(theta)*(XmC)+np.cos(theta)*(YmC))**2/big_radiusy**2),YmC>=0)
	mask_upper= np.multiply((1 <= (np.cos(theta)*(XmC)-np.sin(theta)*(YmC))**2/small_radiusx**2+(np.sin(theta)*(XmC)+np.cos(theta)*(YmC))**2/small_radiusy**2) &(1 >= (np.cos(theta)*(XmC)-np.sin(theta)*(YmC))**2/big_radiusx**2+ (np.sin(theta)*(XmC)+np.cos(theta)*(YmC))**2/big_radiusy**2),YmC<=0)
	if show:
		plt.imshow(mask_lower,interpolation='nearest',cmap='binary')
		plt.show(block=False)
		plt.pause(2)
		plt.close()
		plt.imshow(mask_upper,interpolation='nearest',cmap='binary')
		plt.show(block=False)
		plt.pause(2)
		plt.close()
	if save:
		np.savetxt(destination,mask_lower,delimiter='\t', newline='\n')
	return mask_lower,mask_upper

def getellipseparams(intervals=20,aspectratio=84/49,maxradiusx=738/2):
	#aspect ration =sigmax/sigmay
	brx=np.linspace(maxradiusx/intervals,maxradiusx,num=intervals)
	srx=np.linspace(0, maxradiusx-maxradiusx/intervals, num=intervals)
	bry=brx/aspectratio
	sry=srx/aspectratio
	return brx,bry,srx,sry

def applyannularmaskmulti(filelist,center, intervals=20,aspectratio=84/49,maxradiusx=738/2,theta=0,radians=0,save=1,show=0,destinationsuffix="_er",delimiter="_p*_d",verbose=0,xpixel=738,ypixel=485):
	brx,bry,srx,sry=getellipseparams(intervals,aspectratio=aspectratio,maxradiusx=maxradiusx)
	intervalvals=np.array([], dtype=np.float64).reshape(0,len(filelist))
	for i in range(intervals):
		if verbose:
			print("applying ring mask: "+ str(i) +" from " +str(srx[i])+ "< x < " +str(brx[i]) +" and "+str(sry[i])+ "< y < " +str(bry[i]))
		tm=createannularellipsemask(center, brx[i],bry[i],srx[i],sry[i],theta=theta,xpixel=xpixel,ypixel=ypixel,show=show)
		f_er=getfileprefix(filelist[0],delimiter=".")+destinationsuffix+str(i)+".txt"
		intervalvals=np.vstack((intervalvals,averageROImulti(filelist,tm,save=save,destination=f_er)))
	np.savetxt(getfileprefix(filelist[0],delimiter=".")+destinationsuffix+"_all.txt",intervalvals)
	return brx,bry,srx,sry

def applyhalfannularmaskmulti(filelist,center, intervals=20,aspectratio=84/49,maxradiusx=738/2,theta=0,radians=0,save=1,show=0,destinationsuffix="_er",delimiter="_p*_d",verbose=0,xpixel=738,ypixel=485):
	brx,bry,srx,sry=getellipseparams(intervals,aspectratio=aspectratio,maxradiusx=maxradiusx)
	intervalvals_lower=np.array([], dtype=np.float64).reshape(0,len(filelist))
	intervalvals_upper=np.array([], dtype=np.float64).reshape(0,len(filelist))
	for i in range(intervals):
		if verbose:
			print("applying ring mask: "+ str(i) +" from " +str(srx[i])+ "< x < " +str(brx[i]) +" and "+str(sry[i])+ "< y < " +str(bry[i]))
		tm_lower,tm_upper=createannularhalfellipsemask(center, brx[i],bry[i],srx[i],sry[i],theta=theta,xpixel=xpixel,ypixel=ypixel,show=show)
		f_lower_er=getfileprefix(filelist[0],delimiter=".")+destinationsuffix+"_lower"+str(i)+".txt"
		f_upper_er=getfileprefix(filelist[0],delimiter=".")+destinationsuffix+"_upper"+str(i)+".txt"
		intervalvals_lower=np.vstack((intervalvals_lower,averageROImulti(filelist,tm_lower,save=save,destination=f_lower_er)))
		intervalvals_upper=np.vstack((intervalvals_upper,averageROImulti(filelist,tm_upper,save=save,destination=f_upper_er)))
	np.savetxt(getfileprefix(filelist[0],delimiter=".")+destinationsuffix+"_lower_all.txt",intervalvals_lower)
	np.savetxt(getfileprefix(filelist[0],delimiter=".")+destinationsuffix+"_upper_all.txt",intervalvals_upper)
	return brx,bry,srx,sry

def averageROI(filename,mask):
	"""
	Note if mask has zero elements then denominator will diverge
	"""
	if(np.sum(mask)==0):
		return 0
	datamat=np.loadtxt(filename)
	# if np.shape(datamat)[0]!=np.shape(mask)[0]:
	# 	datamat=np.reshape(datamat,np.shape(mask))
	return np.sum(np.multiply(datamat,mask))/(np.sum(mask))

def averageROImulti(filelist,mask,save=0,destination="kinetictrace.txt"):
	"""
	ie0_ts=ps.averageROImulti(fldata_stretched_ch4,ie0,save=1,destination="ie0_ch4.txt")
	"""
	a=[]
	for f in filelist:
		a.append(averageROI(f,mask))
	if(save):
		np.savetxt(destination,a,delimiter='\t', newline='\n')
	return a

def averagemultiROImulti(filelist,masklist,save=1,destination="kinetictrace.txt"):
	"""
	"""
	mad=np.array([], dtype=np.float64).reshape(0,len(filelist))
	for m in masklist:
		m_fp=getfileprefix(m,delimiter=".")
		mask=np.loadtxt(m)
		f_fp=getfileprefix(filelist[0],delimiter=".")
		mad=np.vstack((mad,averageROImulti(filelist,mask,save=save,destination=f_fp+m_fp+".txt")))
	if save:
		np.savetxt(f_fp+m_fp+"_all.txt",mad,delimiter='\t', newline='\n')
	return mad

def showmask(filelist,maskfilename,show=0,autoclose=0,destinationsuffix="_mask.tif"):
	filelistcleaned=cleanfilelist(filelist,character=".",replacementchar="x")
	for i,f in enumerate(filelist):
		
		fnmask=getfileprefix(filelistcleaned[i],delimiter=".")
		datamat=np.loadtxt(f)
		mask=np.loadtxt(maskfilename)
		datamat_masked=np.multiply(datamat,mask)
		fig1, ax1 = plt.subplots()
		im = ax1.imshow(datamat_masked, interpolation='bilinear', cmap=cm.jet,origin='lower',vmax=np.amax(datamat), vmin=np.amin(datamat))
		cbar=fig1.colorbar(im, ax=ax1)
		cbar.ax.set_ylabel('mJ/cm^2',fontname="Arial")
		plt.savefig(fnmask)
		if show:
			if(autoclose):
				plt.show(block=False)
				plt.pause(2)
				plt.close()
			else:
				plt.show()
		return


def createellipseparametric(center, big_radiusx,big_radiusy,theta=0,num=100,radians=False):
	if(radians==False):
		theta=theta*3.1415/180
	t=np.linspace(0,2*3.1415259,num)
	x=center[0]+big_radiusx*np.cos(t+theta)
	y=center[1]+big_radiusy*np.sin(t)
	return x,y

def createellipseparametricmulti(center, brx,bry,theta=0,num=100,radians=False,save=0,destinationbase="ep_"):
	epallx=np.array([], dtype=float).reshape(0,num)
	epally=np.array([], dtype=float).reshape(0,num)
	for i in range(len(brx)):
		x,y=createellipseparametric(center, brx[i],bry[i],theta=theta,num=num,radians=radians)
		epallx=np.vstack((epallx,x))
		epally=np.vstack((epally,y))
		np.savetxt(destinationbase+str(i)+".txt",np.vstack((x,y)))
	return epallx,epally


def subtractkinetictracemulti(filelist,verbose=0,backgroundposition=-1):
	"""
	Example:
	fl_ts_er_ch4=ps.getfilenames(searchstring0,index=np.arange(20),delimiter="_er",suffix=".txt",verbose=1)
	ps.subtractkinetictracemulti(fl_ts_er_ch4,backgroundposition=-1,verbose=0)
	"""
	background=np.loadtxt(filelist[backgroundposition])
	print(background)
	kt_nb=np.array([], dtype=float).reshape(0,len(background))
	for f in filelist:
		trace=np.loadtxt(f)
		trace_nb=trace-background
		f_nb=generatefilename(f,"_nb",verbose=verbose)
		f_nb_all=generatefilename(f,"_nb_all",verbose=verbose)
		kt_nb=np.vstack((kt_nb,trace_nb))
		np.savetxt(f_nb,trace_nb)
	np.savetxt(f_nb_all,kt_nb)
	return kt_nb

def calcsigmasquared(datamat,shiftx,shifty,mux,muy):
	X,Y=np.meshgrid(shiftx,shifty)
	integrand0=np.multiply((np.power(np.subtract(X,mux),2.0)+np.power(np.subtract(Y,muy),2.0)),datamat)
	n0=np.trapz(np.trapz(integrand0,x=shiftx),x=shifty)
	d0=np.trapz(np.trapz(datamat,x=shiftx),x=shifty)
	return n0/d0/10**8

def calcrsquared(datamat,shiftx,shifty,mux=0,muy=0):
	X,Y=np.meshgrid(shiftx,shifty)
	Xsq=np.power(np.subtract(X,mux-.00000000001),2)
	Ysq=np.power(np.subtract(Y,muy-.00000000001),2)
	Rsq=np.add(Xsq,Ysq)
	d0=np.sum(datamat)+.00000000001
	bRsqb=np.sum(np.multiply(datamat,Rsq))/d0
	bXsqb=np.sum(np.multiply(datamat,Xsq))/d0
	bYsqb=np.sum(np.multiply(datamat,Ysq))/d0
	#bARsqb=(1-bYsqb/(bXsqb+.00000000001))**0.5
	#bepb=beb/(((1-beb**2)**0.5)+.00000000001)
	return bRsqb,bXsqb,bYsqb

def calcrsquaredmulti(filelist,shiftx,shifty,mux=0.0,muy=0.0,save=0,verbose=0):
	kt_rsq=np.array([])
	kt_xsq=np.array([])
	kt_ysq=np.array([])
	#kt_e=np.array([])
	#kt_ep=np.array([])
	for f in filelist:
		if verbose:
			print("calculating r^2 for: ",f) 
		datamat=np.loadtxt(f)
		bRsqb,bXsqb,bYsqb=calcrsquared(datamat,shiftx,shifty,mux,muy)
		kt_rsq=np.concatenate([kt_rsq,[bRsqb]])
		kt_xsq=np.concatenate([kt_xsq,[bXsqb]])
		kt_ysq=np.concatenate([kt_ysq,[bYsqb]])
		# kt_e=np.concatenate([kt_e,[beb]])
		# kt_ep=np.concatenate([kt_ep,[bepb]])
	if save:
		f_rsq=generatefilename(filelist[0],"_kt_rsq",verbose=verbose)
		np.savetxt(f_rsq,kt_rsq)
		f_xsq=generatefilename(filelist[0],"_kt_xsq",verbose=verbose)
		np.savetxt(f_xsq,kt_xsq)
		f_ysq=generatefilename(filelist[0],"_kt_ysq",verbose=verbose)
		np.savetxt(f_ysq,kt_ysq)
		# f_e=generatefilename(filelist[0],"_kt_e",verbose=verbose)
		# np.savetxt(f_e,kt_e)
		# f_ep=generatefilename(filelist[0],"_kt_ep",verbose=verbose)
		# np.savetxt(f_ep,kt_ep)
	return kt_rsq,kt_xsq,kt_ysq

def calcsigmasquaredmulti(filelist,shiftx,shifty,mux=0.0,muy=0.0,save=0,destinationbase="_sigasq",verbose=0):
	kt_sigsq=np.array([], dtype=float).reshape(0,1)
	for f in filelist:
		datamat=np.loadtxt(f)
		kt_sigsq=np.vstack((kt_sigsq,calcsigmasquared(datamat,shiftx,shifty,mux,muy)))
	if save:
		f_sigsq=generatefilename(filelist[0],destinationbase,verbose=verbose)
		np.savetxt(f_sigsq,kt_sigsq)
	return kt_sigsq
#________________________________________________________Line Cuts ________________________________________________________
	
def linecut(filename,lineregion=[], vlinecut=0,show=0,takemax=0,savemask=0):
	"""
	module for performing a line cut acorss an image. It returns an array of dimsion pixels (typically 738 for a horizontal linecut or 485 for a vlinecut)
	:param list float: timepoints: time delays coorespnding to delay stage postion.
	:param list float: lineregion: region in terms of pixels overwhich the line cut is averaged. if default the region is from min pixel to max pixels.
	:param string: filename: file name from with the data matrix is loaded.
	"""
	fn_mask=generatefilename(filename,suffix="_mask",extension="txt",verbose=0)
	datamat=np.loadtxt(filename)
	if(vlinecut):
		if(lineregion==[]):
			if takemax:
				lc=np.amax(datamat,axis=1)
			else:
				lc=np.average(datamat,axis=1)
		else:
			coords=[[lineregion[0],0],[lineregion[1],0],[lineregion[1],np.shape(datamat)[0]],[lineregion[0],np.shape(datamat)[0]]]
			mask=createmask(coords,xpixel=np.shape(datamat)[1],ypixel=np.shape(datamat)[0],show=show,save=savemask)
			if savemask:
				np.savetxt(fn_mask,mask)
			if takemax:
				lc= np.amax(np.multiply(datamat,mask),axis=1)
			else:
				lc= np.average(np.multiply(datamat,mask),axis=1)
	else:
		if(lineregion==[]):
			if takemax:
				lc=np.amax(datamat,axis=0)
			else:
				lc=np.average(datamat,axis=0)
		else:
			coords=[[0,lineregion[0]],[np.shape(datamat)[1],lineregion[0]],[np.shape(datamat)[1],lineregion[1]],[0,lineregion[1]]]
			mask=createmask(coords,xpixel=np.shape(datamat)[1],ypixel=np.shape(datamat)[0],show=show,save=savemask)
			if savemask:
				np.savetxt(fn_mask,mask)
			if takemax:
				lc= np.amax(np.multiply(datamat,mask),axis=0)
			else:
				lc= np.average(np.multiply(datamat,mask),axis=0)
	return lc

def linecutmulti(filelist, lineregion=[],vlinecut=0,show=0,save=0,destination="linecut.txt",header=[],savemask=0):
	#Note the line cuts save file has been transposed. If load hlc use np.transpose(np.loadtxt("linecut.txt"))
	if(np.shape(lineregion)==(2,)):
		lr=np.multiply(np.ones((len(filelist),2)),lineregion)
	else:
		lr=lineregion
	dims=np.shape(np.loadtxt(filelist[0]))
	if(vlinecut):
		b=np.array([], dtype=np.int64).reshape(0,dims[0])
	else:
		b=np.array([], dtype=np.int64).reshape(0,dims[1])
	for i,f in enumerate(filelist):
		print(i)
		b=np.vstack((b,linecut(f,lineregion=lr[i,:],vlinecut=vlinecut,show=show,savemask=savemask)))
	if save:
		if header is not []:
			hstr='\t'.join(header)+'\n'
			np.savetxt(destination, np.transpose(b),delimiter='\t', newline='\n',header=hstr)
		else:
			np.savetxt(destination,np.transpose(b),delimiter='\t', newline='\n')
	return b


"""
=======================================================================================================
SECTION 6 Fitting 
========================================================================================================
"""
def fitlinecutmulti(x,y,p0,alpha=0.05,verbose=0,sp=0,ep=0,save=1,fitfunc='cumulative',destinationbase="linecut_fit",**keyword_parameters):
	"""
	:param float: alpha:  95% confidence interval = 100*(1-alpha)
	Example:
	fit_params,fit_data=ps.fitlinecutmulti(shiftx,hlc,p0,ep=700,verbose=1)
	Example2:
	sp=400
	ep=580
	shiftx,shifty=ps.getpositionaxis(mag0)
	p0=[0.011,2.24,4.53,.709,0.082]
	bounds=([-.05,1.03,3.8,.68,0.015],[.05,2.85,5,.95,0.129])
	hlc_ch3=np.loadtxt("linecut_ch3.txt")
	fit_params_ch3,fit_data_ch3=ps.fitlinecutmulti(shiftx,hlc_ch3,p0,sp=sp,ep=ep,fitfunc='cumulativeplusdgaussdx',verbose=1,bounds=bounds,save=1,destinationbase="linecut_ch3_fit")
	ps.plotlinecut(hlc_ch3[:,sp:ep],axis=shiftx[sp:ep],label=tp,waveoffset=0.0075,fitdata=fit_data_ch3[:,sp:ep],xlabel='position (um)',ylabel='secondary electron intensity (A.U.)')		
	"""
	if ('bounds' in keyword_parameters):
		bounds=keyword_parameters['bounds']
	else:
		bounds=np.ones((2,len(p0)))*np.inf
		bounds[0,:]=-1*bounds[0,:]
		print(bounds)
	if ep==0:
		ep=np.shape(y)[1]-1
	if verbose:
		print("fitting from "+str(x[sp]) +" to "+ str(x[ep]))
	#set up the parameter variables which we will write to
	params=np.array([], dtype=np.int64).reshape(0,len(p0))
	uncertaintiesall=np.array([], dtype=np.int64).reshape(0,len(p0))
	fitdata=np.array([], dtype=np.int64).reshape(0,len(x))
	#calculate student T for confidnece interval
	n = len(x)    # number of data points
	p = len(p0) # number of parameters
	dof = max(0, n - p) # number of degrees of freedom
	tval = t.ppf(1.0-alpha/2., dof) # student-t value for the dof and confidence level
	for i in range(np.shape(y)[0]):
		if(fitfunc=='cumulative'):
			popt, pcov = curve_fit(cumulative, x[sp:ep],y[i,sp:ep],p0=p0,bounds=bounds)
			fitdata=np.vstack((fitdata,cumulative(x,*popt)))
		elif(fitfunc=='cumulativeplusdgaussdx'):
			popt, pcov = curve_fit(cumulativeplusdgaussdx, x[sp:ep],y[i,sp:ep],p0=p0,bounds=bounds)
			fitdata=np.vstack((fitdata,cumulativeplusdgaussdx(x,*popt)))
		elif(fitfunc=='peep_dblexp_conv_interp_b'):
			popt, pcov = curve_fit(peep_dblexp_conv_interp_b, x[sp:ep],y[i,sp:ep],p0=p0,bounds=bounds)
			fitdata=np.vstack((fitdata,peep_dblexp_conv_interp_b(x,*popt)))
		elif(fitfunc=='negtimeconv'):
			popt, pcov = curve_fit(negtimeconv, x[sp:ep],y[i,sp:ep],p0=p0,bounds=bounds)
			fitdata=np.vstack((fitdata,negtimeconv(x,*popt)))
		if verbose:
			print("Fit Parameters of Trace "+ str(i)+": " +str(popt))
		params=np.vstack((params,popt))
		uncertaintiesall = np.vstack((uncertaintiesall,tval*np.sqrt(np.diagonal(pcov))))
	if save:
		np.savetxt(destinationbase+"_popt.txt",params)
		np.savetxt(destinationbase+"_data.txt",fitdata)
		np.savetxt(destinationbase+"_var.txt",uncertaintiesall)
	return params,fitdata

"""
Incomplete
def fitlinecut_global(linecut,shiftx,powers,centercoords,p0a,p0b,boundsa,boundsb,verbose=0):
	fitdataall=np.array([], dtype=float).reshape(0,len(shiftx))
	params=np.array([], dtype=np.int64).reshape(0,len(p0a))
	uncertaintiesall=np.array([], dtype=np.int64).reshape(0,len(p0a))
	fitdata=np.array([], dtype=np.int64).reshape(0,len(shiftx))
	global powers2
	powers2=powers
	global centercoords2
	centercoords2=centercoords
	#bounds=ps.createbounds(p0,boundfactor=1.1)
	for c ,l in enumerate(np1_lc):
	if verbose:
		print("Prosessing: ", f)
		start_time = time.time()
		print("initial guess: ", p0a)
		print("lower bounds: ", boundsa[0])
		print("upper bounds: ", boundsa[1])
	if func=='ccrwrs_conv_interp_global':
		popta, pcova = opt.curve_fit(ccrwrs_conv_interp_global,tp0, data, p0=p0,bounds=bounds,verbose=1)
	if verbose:
		print("--- %s (s) to fit global traces --- " % (time.time() - start_time))
		print("Optimization Parameters: ",popt)
	p0a=np.array([5.88937607e-07, 1.14845595e-03 ,1.94521674e-03,centercoords2[c,1],42,.1])
	p0b=np.array([5.88937607e-07, 1.14845595e-03, 1.94521674e-03,1.87632740e-02])
	boundsa=([0,-.1,0,centercoords2[c,1]-1,41,.00000001],[1,.1,.1,centercoords2[c,1]+20,43,1000])
	boundsb=([0,-.1,0,.00000001],[1,.1,.1,1])
	popta, pcova = opt.curve_fit(expgaussplusline,shiftx, np1_lc[c,:],p0=p0a,bounds=boundsa)
	poptb, pcovb = opt.curve_fit(expgausspluslineheld2,shiftx, np1_lc[c,:],p0=p0b,bounds=boundsb)
	alpha=0.05
	n = len(shiftx)    # number of data points
	p = len(p0b) # number of parameters
	dof = max(0, n - p) # number of degrees of freedom
	tval = t.ppf(1.0-alpha/2., dof)
	uncertaintiesall=np.array([], dtype=float).reshape(0,len(p0b))
	uncertaintiesall = np.vstack((uncertaintiesall,np.multiply(tval,np.sqrt(np.diagonal(pcovb)))))
	print("popta",popta)
	print("poptb",poptb)
	print("95% uncertainty:",uncertaintiesall)
	uncertainties=np.sqrt(np.diagonal(pcov))
	fnfitall=getfileprefix(filelist[0],delimiter="_")+"_all_fit.txt"
	fnopt=getfileprefix(filelist[0],delimiter="_")+"_all_popt.txt"
	fnvar=getfileprefix(filelist[0],delimiter="_")+"_all_pvar.txt"
	np.savetxt(fnfitall,fitdataall)
	np.savetxt(fnopt,popt)
	np.savetxt(fnvar,uncertainties)
	return
"""

def fittimeseries_global(filelist,timepoints,p0,fluence,bounds,rho=0.5,func='ccrwrs_conv_interp_global',verbose=0):
	fitdataall=np.array([], dtype=float).reshape(0,len(timepoints))
	global fluence2
	global rho2
	fluence2=fluence
	rho2=rho
	#bounds=ps.createbounds(p0,boundfactor=1.1)
	data=np.array([], dtype=float).reshape(0,len(timepoints))
	tp0=np.array([], dtype=float).reshape(0,len(timepoints))
	for f in filelist:
		data=np.vstack((data,np.loadtxt(f)))
		tp0=np.vstack((tp0,timepoints))
	data=data.ravel()
	print(data[0:len(timepoints)])
	print(data[-len(timepoints):-1])
	tp0=tp0.ravel()
	if verbose:
		print("Prosessing: ", f)
		start_time = time.time()
		print("initial guess: ", p0)
		print("lower bounds: ", bounds[0])
		print("upper bounds: ", bounds[1])
	if func=='ccrwrs_conv_interp_global':
		popt, pcov = opt.curve_fit(ccrwrs_conv_interp_global,tp0, data, p0=p0,bounds=bounds,verbose=1)
	if verbose:
		print("--- %s (s) to fit global traces --- " % (time.time() - start_time))
		print("Optimization Parameters: ",popt)
	fitdataall=np.reshape(ccrwrs_conv_interp_global(tp0,popt[0],popt[1],popt[2]),(len(fluence2),-1))
	counter=0
	for f in filelist:
		fn=generatefilename(f,suffix="_fit",extension="txt",verbose=verbose)
		np.savetxt(fn,fitdataall[counter])
		counter+=1
	uncertainties=np.sqrt(np.diagonal(pcov))
	fnfitall=getfileprefix(filelist[0],delimiter="_")+"_all_fit.txt"
	fnopt=getfileprefix(filelist[0],delimiter="_")+"_all_popt.txt"
	fnvar=getfileprefix(filelist[0],delimiter="_")+"_all_pvar.txt"
	np.savetxt(fnfitall,fitdataall)
	np.savetxt(fnopt,popt)
	np.savetxt(fnvar,uncertainties)
	return

def fittimeseries(filelist,timepoints,p0,bounds,alpha=.05,func='ccrwrs_conv_interpc',verbose=0):
	optparamsall=np.array([], dtype=float).reshape(0,len(p0))
	uncertaintiesall=np.array([], dtype=float).reshape(0,len(p0))
	fitdataall=np.array([], dtype=float).reshape(0,len(timepoints))
	#calculate student T for confidnece interval
	n = len(timepoints)    # number of data points
	p = len(p0) # number of parameters
	dof = max(0, n - p) # number of degrees of freedom
	tval = t.ppf(1.0-alpha/2., dof) # student-t value for the dof and confidence level
	#bounds=ps.createbounds(p0,boundfactor=1.1)
	for f in filelist:
		data =np.loadtxt(f)
		if verbose:
			print("Prosessing :", f)
			start_time = time.time()
			print("initial guess: ", p0)
			print("lower bounds: ", bounds[0])
			print("upper bounds: ", bounds[1])
		if func=='ccrwrs_conv_interpc':
			popt, pcov = opt.curve_fit(ccrwrs_conv_interpc,timepoints, data, p0=p0,bounds=bounds,verbose=verbose)
			fitdata=ccrwrs_conv_interpc(timepoints,*popt)
		elif( func=='exp_conv_interp'):
			popt, pcov = opt.curve_fit(exp_conv_interp,timepoints, data, p0=p0,bounds=bounds,verbose=verbose)
			fitdata=exp_conv_interp(timepoints,*popt)
		elif(func=='eh_gen_sat_interp'):
			popt, pcov = opt.curve_fit(eh_gen_sat_interp,timepoints, data, p0=p0,bounds=bounds,verbose=verbose)
			fitdata=eh_gen_sat_interp(timepoints,*popt)
		elif(func=='eh_gen_sat_interp_attenuation'):
			popt, pcov = opt.curve_fit(eh_gen_sat_interp_attenuation,timepoints, data, p0=p0,bounds=bounds,verbose=verbose)
			fitdata=eh_gen_sat_interp_attenuation(timepoints,*popt)
		elif(func=='peep_dblexp_conv_interp_b'):
			popt, pcov = opt.curve_fit(peep_dblexp_conv_interp_b, timepoints,data,p0=p0,bounds=bounds,verbose=verbose)
			fitdata=peep_dblexp_conv_interp_b(timepoints,*popt)
		elif(func=='negtimeconv'):
			popt, pcov = opt.curve_fit(negtimeconv, timepoints,data,p0=p0,bounds=bounds,verbose=verbose)
			fitdata=negtimeconv(timepoints,*popt)
		if verbose:
			print("--- %s (s) to fit kinetic trace --- " % (time.time() - start_time))
			print("Optimization Parameters: ",popt)
		fn=generatefilename(f,suffix="_fit",extension="txt",verbose=verbose)
		np.savetxt(fn,fitdata)
		optparamsall=np.vstack((optparamsall,popt))
		uncertaintiesall = np.vstack((uncertaintiesall,np.multiply(tval,np.sqrt(np.diagonal(pcov)))))
		#uncertaintiesall2=[uncertaintiesall[i] if uncertaintiesall[i]<2*optparamsall[i] else 2*optparamsall[i] for i in range(np.shape(uncertaintiesall)[0])]
		#(uncertaintiesall)
		#print(uncertaintiesall2)
		fitdataall=np.vstack((fitdataall,fitdata))
	fnopt=getfileprefix(filelist[0],delimiter="_")+"_all_popt.txt"
	fnvar=getfileprefix(filelist[0],delimiter="_")+"_all_pvar.txt"
	fnfitall=getfileprefix(filelist[0],delimiter="_")+"_all_fit.txt"
	np.savetxt(fnopt,optparamsall)
	np.savetxt(fnfitall,fitdataall)
	np.savetxt(fnvar,uncertaintiesall)
	return

# def fittimeserieshold(filelist,timepoints,p0,hold,bounds,alpha=.05,func='ccrwrs_conv_interpc',verbose=0):
# 	optparamsall=np.array([], dtype=float).reshape(0,len(p0))
# 	uncertaintiesall=np.array([], dtype=float).reshape(0,len(p0))
# 	fitdataall=np.array([], dtype=float).reshape(0,len(timepoints))
# 	#calculate student T for confidnece interval
# 	n = len(timepoints)    # number of data points
# 	p = len(p0) # number of parameters
# 	dof = max(0, n - p) # number of degrees of freedom
# 	tval = t.ppf(1.0-alpha/2., dof) # student-t value for the dof and confidence level
# 	#bounds=ps.createbounds(p0,boundfactor=1.1)
# 	#p1=[p0[i] for i in range(len(hold)) if hold==1]
# 	holdloc=[i for i, x in enumerate(hold) if x]
# 	p1=np.delete(p0,holdloc)
# 	global hold2
# 	hold2=hold
# 	global p2
# 	p2=p0
# 	bounds1=[[bounds[0,i],for i in range(len(hold)) if hold==1],[bounds[1,i],for i in range(len(hold)) if hold==1]]
# 	cmdfunction='lambda timepoints, *p: ccrwrs_conv_interpc(timepoints,)'
# 	function=
# 	cmdfit='opt.curve_fit({},timepoints,data,p0=p1,bounds=bounds1,verbose=verbose'.format(func)
# 	popt, pcov=eval(cmdfit)
# 	for f in filelist:
# 		data =np.loadtxt(f)
# 		if verbose:
# 			print("Prosessing :", f)
# 			start_time = time.time()
# 			print("initial guess: ", p0)
# 			print("lower bounds: ", bounds[0])
# 			print("upper bounds: ", bounds[1])
# 			popt, pcov=eval(cmdfit)
# 			fitdata=np.vstack((fitdata,peep_dblexp_conv_interp_b(timepoints,*popt)))
# 		if verbose:
# 			print("--- %s (s) to fit kinetic trace --- " % (time.time() - start_time))
# 			print("Optimization Parameters: ",popt)
# 		fn=generatefilename(f,suffix="_fit",extension="txt",verbose=verbose)
# 		np.savetxt(fn,fitdata)
# 		optparamsall=np.vstack((optparamsall,popt))
# 		uncertaintiesall = np.vstack((uncertaintiesall,np.multiply(tval,np.sqrt(np.diagonal(pcov)))))
# 		fitdataall=np.vstack((fitdataall,fitdata))
# 	fnopt=getfileprefix(filelist[0],delimiter="_")+"_all_popt.txt"
# 	fnvar=getfileprefix(filelist[0],delimiter="_")+"_all_pvar.txt"
# 	fnfitall=getfileprefix(filelist[0],delimiter="_")+"_all_fit.txt"
# 	np.savetxt(fnopt,optparamsall)
# 	np.savetxt(fnfitall,fitdataall)
# 	np.savetxt(fnvar,uncertaintiesall)
# 	return
#________________________________________________________1D Time Series_______________________________________________________

def ccrwrs(t,a,l,tau1,tau2,sigma):
	# timepoints=np.linspace(-500,7000,1000)
	# f=ps.ccrwrs_conv(timepoints,1,10000000,140,90000,10)
	# ps.plotkinetictrace(f,timepoints,show=1,autoclose=0,logx=0,save=1,destination="timeseries.txt")
	gauss=np.exp(-(t/sigma)**2/2)/sigma
	s=a*(1-np.exp(-l))*(np.exp(-t/tau1)-np.exp(-t/tau2))*np.heaviside(t, 0)
	return s

def ccrwrs_conv_lin(t,a,l,tau1,tau2,sigma):
	#NOTE ONLY WORKS FOR LINEARLY SPACE COORDINATES IN t
	# timepoints=np.linspace(-500,7000,1000)
	# f=ps.ccrwrs_conv(timepoints,1,10000000,140,90000,10)
	# ps.plotkinetictrace(f,timepoints,show=1,autoclose=0,logx=0,save=1,destination="timeseries.txt")
	if t[-1]<t[0]:
		print("XXXXX ERRROR XXXXX time array must be increasing for convolve to work")
	gauss=np.exp(-(t/sigma)**2/2)/sigma
	s =a*(1-np.exp(-l))*(np.exp(-t/tau1)-np.exp(-t/tau2))*np.heaviside(t, 0)
	ssgauss=np.convolve(s, gauss, mode="full")[0:len(t)]
	return ssgauss

def ccrwrs_conv_linb(t,a,l,t0,tau1,tau2,sigma):
	#NOTE ONLY WORKS FOR LINEARLY SPACE COORDINATES IN t
	# timepoints=np.linspace(-500,7000,1000)
	# f=ps.ccrwrs_conv(timepoints,1,10000000,140,90000,10)
	# ps.plotkinetictrace(f,timepoints,show=1,autoclose=0,logx=0,save=1,destination="timeseries.txt")
	if t[-1]<t[0]:
		print("XXXXX ERRROR XXXXX time array must be increasing for convolve to work")
	gauss=np.exp(-((t-t0)/sigma)**2/2)/sigma
	s =a*(1-np.exp(-l))*(np.exp(-(t-t0)/tau1)-np.exp(-(t-t0)/tau2))*np.heaviside((t-t0), 0)
	ssgauss=np.convolve(s, gauss, mode="full")[0:len(t)]
	return ssgauss

def ccrwrs_conv(t,a,l,tau1,tau2,sigma):
	# timepoints=np.linspace(-500,7000,1000)
	# f=ps.ccrwrs_conv(timepoints,1,10000000,140,90000,10)
	# ps.plotkinetictrace(f,timepoints,show=1,autoclose=0,logx=0,save=1,destination="timeseries.txt")
	if t[-1]<t[0]:
		print("XXXXX ERRROR XXXXX time array must be increasing for convolve to work")
	t2=np.linspace(t[0],t[-1],5000)
	gauss=np.exp(-(t2/sigma)**2/2)/sigma
	s =a*(1-np.exp(-l))*(np.exp(-t2/tau1)-np.exp(-t2/tau2))*np.heaviside(t2, 0)
	ssgauss=np.convolve(s, gauss, mode="full")[0:len(t2)]
	idx=[(np.abs(t2 - tx)).argmin() for tx in t]
	s2=[ssgauss[i] for i in idx]
	return s2

def ccrwrs_conv_interp(t,a,l,tau1,tau2,sigma):
	"""
	import matplotlib.pyplot as plt
	tp0=ps.loadtimepoints("timepoints20200131181703.txt")
	timepoints=np.linspace(tp0[0],tp0[-1],5000)
	p0=[-.08,10000000,150,700,10]
	f=ps.ccrwrs_conv_lin(timepoints,-.08,10000000,150,7000,1000)
	f2=ps.ccrwrs_conva(tp0,-.08,10000000,150,7000,1000)
	f3=ps.ccrwrs_conv_interp(tp0,-.08,10000000,150,7000,1000)
	if 1:
		plt.plot(timepoints, f,tp0, f2,tp0, f3, marker='o',)
		plt.show()
	"""
	if t[-1]<t[0]:
		print("XXXXX ERRROR XXXXX time array must be increasing for convolve to work")
	t2=np.linspace(t[0],t[-1],5000)
	gauss=np.exp(-(t2/sigma)**2/2)/sigma
	s =a*(1-np.exp(-l))*(np.exp(-t2/tau1)-np.exp(-t2/tau2))*np.heaviside(t2, 0)
	ssgauss=np.convolve(s, gauss, mode="full")[0:len(t2)]
	idx=[(np.abs(t2 - tx)).argmin() for tx in t]
	pm=[np.sign(t2[idx[j]]-t[j]) for j in range(len(idx))]
	s3=[]
	#linearly interpolate ssgauss between interval of t2
	for i in range(len(idx)):
		if pm[i]==0:
			s3.append(ssgauss[idx[i]])
		elif pm[i]<0:
			s3.append(ssgauss[idx[i]-1]+(ssgauss[idx[i]]-ssgauss[idx[i]-1])*(t[i]-t2[idx[i]-1])/(t2[idx[i]]-t2[idx[i]-1]))
		elif pm[i]>0:
			s3.append(ssgauss[idx[i]]+(ssgauss[idx[i]+1]-ssgauss[idx[i]])*(t[i]-t2[idx[i]])/(t2[idx[i]+1]-t2[idx[i]]))
	return s3

def ccrwrs_conv_interpc(t,a,t0,tau1,tau2,sigma=15.4,rangefactor=0.4977255):
	"""
	import matplotlib.pyplot as plt
	import numpy as np
	import pysuem as ps
	tp0=np.flip(ps.loadtimepoints("timepoints1.txt"))
	timepoints=np.linspace(tp0[0],tp0[-1],46200)
	if 1:
		f5=ps.ccrwrs_conv_interpc(tp0,-.08,10,50,5000,1.0)
		f6=ps.ccrwrs_conv_interpc(timepoints,-.08,10,50,5000,1.0)
		#plt.xlim(-50,50)
		plt.plot(tp0, f5,timepoints,f6 ,marker='o',label="3")
		plt.legend(loc='best')
		plt.show()
	"""
	if t[-1]<t[0]:
		print("XXXXX ERRROR XXXXX time array must be increasing for convolve to work")
	numpoints= 46200
	#buffer the time domain to be twice as big so we don't get edge effects.
	mid=(t[-1]+t[0])/2
	r=t[-1]-t[0]
	t2=np.linspace(mid-r,mid+r,numpoints)
	gauss=np.exp(-((t2)/sigma)**2/2)/sigma
	s =a*(np.exp(-(t2-t0)/tau1)-np.exp(-(t2-t0)/tau2))*np.heaviside((t2-t0), 0)
	ssgauss=np.convolve(s, gauss, mode="full")[int(numpoints*rangefactor):int(numpoints*(rangefactor+1))]
	idx=[(np.abs(t2 - tx)).argmin() for tx in t]
	pm=[np.sign(t2[idx[j]]-t[j]) for j in range(len(idx))]
	s3=[]
	#linearly interpolate ssgauss between interval of t2
	for i in range(len(idx)):
		if pm[i]==0:
			s3.append(ssgauss[idx[i]])
		elif pm[i]<0:
			s3.append(ssgauss[idx[i]-1]+(ssgauss[idx[i]]-ssgauss[idx[i]-1])*(t[i]-t2[idx[i]-1])/(t2[idx[i]]-t2[idx[i]-1]))
		elif pm[i]>0:
			s3.append(ssgauss[idx[i]]+(ssgauss[idx[i]+1]-ssgauss[idx[i]])*(t[i]-t2[idx[i]])/(t2[idx[i]+1]-t2[idx[i]]))
	return s3

def exp_conv_lin(t,a,tau1,sigma,t0):
	"""
	"""
	if t[-1]<t[0]:
		print("XXXXX ERRROR XXXXX time array must be increasing for convolve to work")
	gauss=np.exp(-((t)/sigma)**2/2)/sigma
	s =(a*np.exp(-(t-t0)/tau1))*np.heaviside((t-t0), 0)
	ssgauss=np.convolve(s, gauss, mode="full")[int(len(t)*1/2):int(len(t)*3/2)]
	return ssgauss

def exp_conv_linb(t,a,tau1,sigma,t0):
	"""
	"""
	if t[-1]<t[0]:
		print("XXXXX ERRROR XXXXX time array must be increasing for convolve to work")
	mid=(t[-1]+t[0])/2
	r=t[-1]-t[0]
	t3=np.linspace(mid-r/2,mid+r/2,2*len(t))
	gauss=np.exp(-((t3)/sigma)**2/2)/sigma
	s =(np.exp(-(t3-t0)/tau1))*np.heaviside((t3-t0), 0)
	rangefactor=2.972745/4
	ssgauss=np.convolve(s, gauss, mode="full")[int(len(t3)*rangefactor):int(len(t3)*(rangefactor+1)):2]
	return np.multiply(ssgauss,a)


def expt0(t,a,tau1,t0):
	return a*(np.exp(-(t-t0)/tau1))*np.heaviside((t-t0), 0)

def exp_conv_linc(t,a,tau1,sigma,t0,rangefactor=.747725):
	"""
	"""
	if t[-1]<t[0]:
		print("XXXXX ERRROR XXXXX time array must be increasing for convolve to work")
	mid=(t[-1]+t[0])/2
	r=t[-1]-t[0]
	t3=np.linspace(mid-r,mid+r,2*len(t))
	gauss=np.exp(-((t3)/sigma)**2/2)/sigma
	s =(np.exp(-(t3-t0)/tau1))*np.heaviside((t3-t0), 0)
	ssgauss=np.convolve(s, gauss, mode="full")[int(len(s)*rangefactor):int(len(s)*(rangefactor+1/2))]
	return np.multiply(ssgauss,a/2)


def exp_conv_interpb(t,a,tau1,sigma,t0,rangefactor=0.4977255):
	"""
	"""
	if t[-1]<t[0]:
		print("XXXXX ERRROR XXXXX time array must be increasing for convolve to work")
	numpoints= 46200
	#buffer the time domain to be twice as big so we don't get edge effects.
	mid=(t[-1]+t[0])/2
	r=t[-1]-t[0]
	t2=np.linspace(mid-r,mid+r,numpoints)
	gauss=np.exp(-((t2)/sigma)**2/2)/sigma
	s =(np.exp(-(t2-t0)/tau1))*np.heaviside((t2-t0), 0)
	ssgauss=np.convolve(s, gauss, mode="full")[int(numpoints*rangefactor):int(numpoints*(rangefactor+1))]
	idx=[(np.abs(t2 - tx)).argmin() for tx in t]
	pm=[np.sign(t2[idx[j]]-t[j]) for j in range(len(idx))]
	s3=[]
	#linearly interpolate ssgauss between interval of t2
	for i in range(len(idx)):
		if pm[i]==0:
			s3.append(ssgauss[idx[i]])
		elif pm[i]<0:
			s3.append(ssgauss[idx[i]-1]+(ssgauss[idx[i]]-ssgauss[idx[i]-1])*(t[i]-t2[idx[i]-1])/(t2[idx[i]]-t2[idx[i]-1]))
		elif pm[i]>0:
			s3.append(ssgauss[idx[i]]+(ssgauss[idx[i]+1]-ssgauss[idx[i]])*(t[i]-t2[idx[i]])/(t2[idx[i]+1]-t2[idx[i]]))
	return np.multiply(s3,a)


def dblexp_conv_interp(t,a1,tau1,a2,tau2,sigma,t0):
	"""
	"""
	
	numpoints=10000
	t2=np.linspace(t[0],t[-1],numpoints)
	gauss=np.exp(-((t2-t0)/sigma)**2/2)/sigma
	s =(a1*np.exp(-(t2-t0)/tau1)+a2*np.exp(-(t2-t0)/tau2))*np.heaviside((t2-t0), 0)
	ssgauss=np.convolve(s, gauss, mode="full")[0:len(t2)]
	idx=[(np.abs(t2 - tx)).argmin() for tx in t]
	pm=[np.sign(t2[idx[j]]-t[j]) for j in range(len(idx))]
	s3=[]
	#linearly interpolate ssgauss between interval of t2
	for i in range(len(idx)):
		if pm[i]==0:
			s3.append(ssgauss[idx[i]])
		elif pm[i]<0:
			s3.append(ssgauss[idx[i]-1]+(ssgauss[idx[i]]-ssgauss[idx[i]-1])*(t[i]-t2[idx[i]-1])/(t2[idx[i]]-t2[idx[i]-1]))
		elif pm[i]>0:
			s3.append(ssgauss[idx[i]]+(ssgauss[idx[i]+1]-ssgauss[idx[i]])*(t[i]-t2[idx[i]])/(t2[idx[i]+1]-t2[idx[i]]))
	return s3

def exp_conv_interpc(t,a,tau1,tau2):
	"""
	import matplotlib.pyplot as plt
	tp0=ps.loadtimepoints("timepoints20200131181703.txt")
	timepoints=np.linspace(tp0[0],tp0[-1],5000)
	p0=[-.08,10000000,150,700,10]
	f=ps.ccrwrs_conv_lin(timepoints,-.08,10000000,150,7000,1000)
	f2=ps.ccrwrs_conva(tp0,-.08,10000000,150,7000,1000)
	f3=ps.ccrwrs_conv_interp(tp0,-.08,10000000,150,7000,1000)
	if 1:
		plt.plot(timepoints, f,tp0, f2,tp0, f3, marker='o',)
		plt.show()
	"""
	t0=-660.
	sigma=9.2*2**.5
	numpoints=5000
	t2=np.linspace(t[0],t[-1],numpoints)
	gauss=np.exp(-((t2-t0)/sigma)**2/2)/sigma
	s =a*(np.exp(-(t2-t0)/tau1)-np.exp(-(t2-t0)/tau2))*np.heaviside((t2-t0), 0)
	ssgauss=np.convolve(s, gauss, mode="full")[0:len(t2)]
	idx=[(np.abs(t2 - tx)).argmin() for tx in t]
	pm=[np.sign(t2[idx[j]]-t[j]) for j in range(len(idx))]
	s3=[]
	#linearly interpolate ssgauss between interval of t2
	for i in range(len(idx)):
		if pm[i]==0:
			s3.append(ssgauss[idx[i]])
		elif pm[i]<0:
			s3.append(ssgauss[idx[i]-1]+(ssgauss[idx[i]]-ssgauss[idx[i]-1])*(t[i]-t2[idx[i]-1])/(t2[idx[i]]-t2[idx[i]-1]))
		elif pm[i]>0:
			s3.append(ssgauss[idx[i]]+(ssgauss[idx[i]+1]-ssgauss[idx[i]])*(t[i]-t2[idx[i]])/(t2[idx[i]+1]-t2[idx[i]]))
	return s3

def isat(y,t):
	a=y[0]
	b=y[1]
	bsat=y[2]
	tau1=y[3]
	tau2=y[4]
	#bsat=1
	#tau1=120
	#tau2=10700
	da=-1/tau1*a+1/tau1*a*b/bsat-1/tau2*a
	db=1/tau1*a-1/tau1*a*b/bsat-1/tau2*b
	#da=-k1*a-k2*a
	#db=k1*a-k3*b
	return [da,db,0,0,0]

def ehsat(y,t,tau1,esat,tau2,hsat):
	eb=y[0]
	edr=y[1]
	hb=y[2]
	hdr=y[3]
	#deb=(hnu0/sigma0/(2*np.pi())**2)*np.exp(-(t-t0)**2/2/sigma0**2)-1/tau1*(1+edr/esat)*eb
	deb=-1/tau1*(1-edr/esat)*eb
	dedr=1/tau1*(1-edr/esat)*eb
	#dhb=(hnu0/sigma0/(2*np.pi())**2)*np.exp(-(t-t0)**2/2/sigma0**2)-1/tau2*(1+hdr/hsat)*hb
	dhb=-1/tau2*(1-hdr/hsat)*hb
	dhdr=1/tau2*(1-hdr/hsat)*hb
	return [deb,dedr,dhb,dhdr]

def dgdt(y,t,t0,sigma0):
	t2=y[0]
	g=y[1]
	dt=t-t2
	#dg=0
	dg=1/(sigma0*(2*np.pi)**.5)*np.exp(-(t2-t0)**2/2/sigma0**2)
	return [dt,dg]

def eh_gen_sat(y,t,hnu0,t0,sigma0,tau1,esat,tau2,hsat):
	"""
	t=np.linspace(0,7000,20000)
	powerseries= np.logspace(-1,4, num=11,base=np.e)
	fig, ax = plt.subplots()
	colors = [plt.cm.jet(i) for i in np.linspace(0, 1,len(powerseries))]
	ax.set_prop_cycle('color', colors)
	for i in powerseries:
		label0=str(i)[0:4]
		#hnu0,t0,sigma0,tau1,esat,tau2,hsat
		b =odeint(ps.eh_gen_sat, [0,0,0,0,0,0], t,args=(i,1000.,100.,350,1,10000,.5))
		plt.plot(t,b[:,3]-b[:,5],label=label0)
	plt.rcParams.update({'font.size': 10})  # increase the font size
	plt.xlabel("time Delay (ps")
	plt.ylabel("Photovoltage [e_\{dr\}]-[h_\{dr\}]")
	plt.legend(loc='best')
	#plt.xlim(-202,-198)
	#plt.xticks([10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000]) 
	plt.show()
	"""
	#note t must start at 0
	t2=y[0]
	g=y[1]
	eb=y[2]
	edr=y[3]
	hb=y[4]
	hdr=y[5]
	#dg=0
	dt=t-t2
	dg=hnu0/(sigma0*(2*np.pi)**.5)*np.exp(-(t2-t0)**2/2/sigma0**2)
	deb=dg-1/tau1*(1-edr/esat)*eb
	dedr=1/tau1*(1-edr/esat)*eb
	dhb=dg-1/tau2*(1-hdr/hsat)*hb
	dhdr=1/tau2*(1-hdr/hsat)*hb
	return [dt,dg,deb,dedr,dhb,dhdr]


def e_gen_sat(y,t,hnu0,t0,sigma0,tau1,esat):
	"""
	t=np.linspace(0,7000,20000)
	powerseries= np.logspace(-1,4, num=11,base=np.e)
	fig, ax = plt.subplots()
	colors = [plt.cm.jet(i) for i in np.linspace(0, 1,len(powerseries))]
	ax.set_prop_cycle('color', colors)
	for i in powerseries:
		label0=str(i)[0:4]
		#hnu0,t0,sigma0,tau1,esat,tau2,hsat
		b =odeint(ps.eh_gen_sat, [0,0,0,0,0,0], t,args=(i,1000.,100.,350,1,10000,.5))
		plt.plot(t,b[:,3]-b[:,5],label=label0)
	plt.rcParams.update({'font.size': 10})  # increase the font size
	plt.xlabel("time Delay (ps")
	plt.ylabel("Photovoltage [e_\{dr\}]-[h_\{dr\}]")
	plt.legend(loc='best')
	#plt.xlim(-202,-198)
	#plt.xticks([10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000]) 
	plt.show()
	"""
	#note t must start at 0
	t2=y[0]
	g=y[1]
	eb=y[2]
	edr=y[3]
	#dg=0
	dt=t-t2
	dg=hnu0/(sigma0*(2*np.pi)**.5)*np.exp(-(t2-t0)**2/2/sigma0**2)
	deb=dg-1/tau1*(1-edr/esat)*eb
	dedr=1/tau1*(1-edr/esat)*eb
	return [dt,dg,deb,dedr]

def e_gen_sat2(y,t,hnu0,t0,sigma0,tau1,tau2,esat):
	"""
	t=np.linspace(0,7000,20000)
	powerseries= np.logspace(-1,4, num=11,base=np.e)
	fig, ax = plt.subplots()
	colors = [plt.cm.jet(i) for i in np.linspace(0, 1,len(powerseries))]
	ax.set_prop_cycle('color', colors)
	for i in powerseries:
		label0=str(i)[0:4]
		#hnu0,t0,sigma0,tau1,esat,tau2,hsat
		b =odeint(ps.eh_gen_sat, [0,0,0,0,0,0], t,args=(i,1000.,100.,350,1,10000,.5))
		plt.plot(t,b[:,3]-b[:,5],label=label0)
	plt.rcParams.update({'font.size': 10})  # increase the font size
	plt.xlabel("time Delay (ps")
	plt.ylabel("Photovoltage [e_\{dr\}]-[h_\{dr\}]")
	plt.legend(loc='best')
	#plt.xlim(-202,-198)
	#plt.xticks([10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000]) 
	plt.show()
	"""
	#note t must start at 0
	t2=y[0]
	g=y[1]
	eb=y[2]
	edr=y[3]
	#dg=0
	dt=t-t2
	dg=hnu0/(sigma0*(2*np.pi)**.5)*np.exp(-(t2-t0)**2/2/sigma0**2)
	deb=dg-1/tau1*(1-edr/esat)*eb
	dedr=1/tau1*(1-edr/esat)*eb-1/tau2*edr
	return [dt,dg,deb,dedr]

def e_gen_sat3(y,t,hnu0,t0,sigma0,tau1,tau2,esat,gamma,alpha):
	"""
	t=np.linspace(0,7000,20000)
	powerseries= np.logspace(-1,4, num=11,base=np.e)
	fig, ax = plt.subplots()
	colors = [plt.cm.jet(i) for i in np.linspace(0, 1,len(powerseries))]
	ax.set_prop_cycle('color', colors)
	for i in powerseries:
		label0=str(i)[0:4]
		#hnu0,t0,sigma0,tau1,esat,tau2,hsat
		b =odeint(ps.eh_gen_sat, [0,0,0,0,0,0], t,args=(i,1000.,100.,350,1,10000,.5))
		plt.plot(t,b[:,3]-b[:,5],label=label0)
	plt.rcParams.update({'font.size': 10})  # increase the font size
	plt.xlabel("time Delay (ps")
	plt.ylabel("Photovoltage [e_\{dr\}]-[h_\{dr\}]")
	plt.legend(loc='best')
	#plt.xlim(-202,-198)
	#plt.xticks([10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000]) 
	plt.show()
	"""
	#note t must start at 0
	t2=y[0]
	g=y[1]
	eb=y[2]
	edr=y[3]
	#dg=0
	dt=t-t2
	dg=hnu0/(sigma0*(2*np.pi)**.5)*np.exp(-(t2-t0)**2/2/sigma0**2)
	deb=dg-1/tau1*(1-edr/esat)*eb
	dedr=1/tau1*(1-edr/esat)*eb-1/tau2*edr*(1+gamma*edr)**alpha
	return [dt,dg,deb,dedr]

def e_gen_sat4(y,t,hnu0,t0,sigma0,tau1,tau2,tau3,esat,gamma,alpha,xi):
	"""
	t=np.linspace(0,7000,20000)
	powerseries= np.logspace(-1,4, num=11,base=np.e)
	fig, ax = plt.subplots()
	colors = [plt.cm.jet(i) for i in np.linspace(0, 1,len(powerseries))]
	ax.set_prop_cycle('color', colors)
	for i in powerseries:
		label0=str(i)[0:4]
		#hnu0,t0,sigma0,tau1,esat,tau2,hsat
		b =odeint(ps.eh_gen_sat, [0,0,0,0,0,0], t,args=(i,1000.,100.,350,1,10000,.5))
		plt.plot(t,b[:,3]-b[:,5],label=label0)
	plt.rcParams.update({'font.size': 10})  # increase the font size
	plt.xlabel("time Delay (ps")
	plt.ylabel("Photovoltage [e_\{dr\}]-[h_\{dr\}]")
	plt.legend(loc='best')
	#plt.xlim(-202,-198)
	#plt.xticks([10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000]) 
	plt.show()
	"""
	#note t must start at 0
	t2=y[0]
	g=y[1]
	eb=y[2]
	escr=y[2]
	etrap=y[3]
	#dg=0
	dt=t-t2
	dg=hnu0/(sigma0*(2*np.pi)**.5)*np.exp(-(t2-t0)**2/2/sigma0**2)
	deb=dg-1/tau1*eb+1/tau1*escr
	dscr= 1/tau1*eb-1/tau1*escr-1/tau3*(etrap-esat)*xi*escr
	dtrap=1/tau3*(etrap-esat)*xi-1/tau2*etrap*(1+gamma*etrap)**alpha
	return [dt,dg,deb,descr,dtrap]

def e_gen_sat5(y,t,hnu0,t0,sigma0,tau1,tau2,tau3,esat,gamma,alpha,xi):
	"""
	t=np.linspace(0,7000,20000)
	powerseries= np.logspace(-1,4, num=11,base=np.e)
	fig, ax = plt.subplots()
	colors = [plt.cm.jet(i) for i in np.linspace(0, 1,len(powerseries))]
	ax.set_prop_cycle('color', colors)
	for i in powerseries:
		label0=str(i)[0:4]
		#hnu0,t0,sigma0,tau1,esat,tau2,hsat
		b =odeint(ps.eh_gen_sat, [0,0,0,0,0,0], t,args=(i,1000.,100.,350,1,10000,.5))
		plt.plot(t,b[:,3]-b[:,5],label=label0)
	plt.rcParams.update({'font.size': 10})  # increase the font size
	plt.xlabel("time Delay (ps")
	plt.ylabel("Photovoltage [e_\{dr\}]-[h_\{dr\}]")
	plt.legend(loc='best')
	#plt.xlim(-202,-198)
	#plt.xticks([10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000]) 
	plt.show()
	"""
	#note t must start at 0
	t2=y[0]
	g=y[1]
	eb=y[2]
	esurface=y[2]
	etrap=y[3]
	#dg=0
	dt=t-t2
	dg=hnu0/(sigma0*(2*np.pi)**.5)*np.exp(-(t2-t0)**2/2/sigma0**2)
	deb=dg-2/tau1*eb+1/tau1*esurface
	dsurface= 1/tau1*eb-1/tau1*esurface-1/tau3*(etrap-esat)*xi*esurface
	dtrap=1/tau3*(etrap-esat)*xi-1/tau2*etrap*(1+gamma*etrap)**alpha
	return [dt,dg,deb,desurface,dtrap]

def e_gen_sat6(y,t,hnu0,t0,sigma0,tau1,tau2,esat,gamma,alpha):
	"""
	t=np.linspace(0,7000,20000)
	powerseries= np.logspace(-1,4, num=11,base=np.e)
	fig, ax = plt.subplots()
	colors = [plt.cm.jet(i) for i in np.linspace(0, 1,len(powerseries))]
	ax.set_prop_cycle('color', colors)
	for i in powerseries:
		label0=str(i)[0:4]
		#hnu0,t0,sigma0,tau1,esat,tau2,hsat
		b =odeint(ps.eh_gen_sat, [0,0,0,0,0,0], t,args=(i,1000.,100.,350,1,10000,.5))
		plt.plot(t,b[:,3]-b[:,5],label=label0)
	plt.rcParams.update({'font.size': 10})  # increase the font size
	plt.xlabel("time Delay (ps")
	plt.ylabel("Photovoltage [e_\{dr\}]-[h_\{dr\}]")
	plt.legend(loc='best')
	#plt.xlim(-202,-198)
	#plt.xticks([10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000]) 
	plt.show()
	"""
	#note t must start at 0
	t2=y[0]
	g=y[1]
	eb=y[2]
	edr=y[3]
	#dg=0
	dt=t-t2
	dg=hnu0/(sigma0*(2*np.pi)**.5)*np.exp(-(t2-t0)**2/2/sigma0**2)
	deb=dg-1/tau1*(1-edr/esat)*eb-1/tau1*eb
	dedr=1/tau1*(1-edr/esat)*eb-1/tau2*edr*(1+gamma*edr)**alpha
	return [dt,dg,deb,dedr]

def e_gen_sat7(y,t,hnu0,t0,sigma0,tau1,tau2,esat,gamma):
	"""
	t=np.linspace(0,7000,20000)
	powerseries= np.logspace(-1,4, num=11,base=np.e)
	fig, ax = plt.subplots()
	colors = [plt.cm.jet(i) for i in np.linspace(0, 1,len(powerseries))]
	ax.set_prop_cycle('color', colors)
	for i in powerseries:
		label0=str(i)[0:4]
		#hnu0,t0,sigma0,tau1,esat,tau2,hsat
		b =odeint(ps.eh_gen_sat, [0,0,0,0,0,0], t,args=(i,1000.,100.,350,1,10000,.5))
		plt.plot(t,b[:,3]-b[:,5],label=label0)
	plt.rcParams.update({'font.size': 10})  # increase the font size
	plt.xlabel("time Delay (ps")
	plt.ylabel("Photovoltage [e_\{dr\}]-[h_\{dr\}]")
	plt.legend(loc='best')
	#plt.xlim(-202,-198)
	#plt.xticks([10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000]) 
	plt.show()
	"""
	#note t must start at 0
	t2=y[0]
	g=y[1]
	eb=y[2]
	edr=y[3]
	#dg=0
	dt=t-t2
	dg=hnu0/(sigma0*(2*np.pi)**.5)*np.exp(-(t2-t0)**2/2/sigma0**2)
	deb=dg-1/tau1*(1-edr/esat)*eb-1/tau1*eb
	dedr=1/tau1*(1-edr/esat)*eb-1/tau2*edr*(1+gamma*edr)
	return [dt,dg,deb,dedr]

def esat_expgaussdoubleconv(t,hnu0,scalar,t0,sigma0,tau1,tau2,tau3,esat,gamma,alpha):
	if t[-1]<t[0]:
		print("XXXXX ERRROR XXXXX time array must be increasing odeint to work.")
	#buffer the time domain to be twice as big so we don't get edge effects.
	t2=np.linspace(-10000,9999,20000)
	t3=t2-t2[0]
	#idt0=(np.abs(t2 - t0)).argmin()
	s =odeint(e_gen_sat6, [0,0,0,0], t3,args=(hnu0,t0+t3[0]-t2[0],sigma0,tau1,tau2,esat,gamma,alpha),hmax=1.0)
	exp=np.exp(((t2)/tau3))*np.heaviside((-t2+t0), 0)/tau3
	gauss=np.exp(-((t2-t0)/sigma0)**2/2)/(2*3.14152*sigma0)
	#ps.plotkinetictrace(exp,t2,save=1,destination="testhevix.txt",logx=0)
	#ps.plotkinetictrace(s[:,3],t2,save=1,destination="testhevix.txt",logx=0)
	#print("xxxxxxx")
	#print(len(exp))
	#print(type(exp))
	#print(np.shape(s[:,3]))
	#print(type(s))
	ssexp=np.convolve(s[:,3], exp, mode="full")
	ssgauss=np.convolve(s[:,3], gauss, mode="full")
	s2=scalar*ssexp[9999:29999]-1*ssgauss[9999:29999]
	#print(np.shape(ssexp[9999:29999]))
	#ps.plotkinetictrace(ssexp[9999:29999],t2,save=1,destination="testhevix.txt",logx=0)
	#print("xxxxxxx")
	idx=[(np.abs(t2 - tx)).argmin() for tx in t]
	pm=[np.sign(t2[idx[j]]) for j in range(len(idx))]
	#print("idx:",idx)
	#print("t",t)
	#print("t2[idx]",t2[idx])
	#print("pm:",pm)
	s3=[]
	#linearly interpolate ssexp between interval of t2
	for i in range(len(idx)):
		if pm[i]==0:
			s3.append(s2[idx[i]])
		elif pm[i]<0:
			s3.append(s2[idx[i]-1])
		elif pm[i]>0:
			s3.append(s2[idx[i]])
	return s3

def esat_expconv(t,hnu0,scalar,t0,sigma0,tau1,tau2,tau3,esat,gamma,alpha):
	if t[-1]<t[0]:
		print("XXXXX ERRROR XXXXX time array must be increasing odeint to work.")
	#buffer the time domain to be twice as big so we don't get edge effects.
	t2=np.linspace(-10000,9999,20000)
	t3=t2-t2[0]
	#idt0=(np.abs(t2 - t0)).argmin()
	s =odeint(e_gen_sat6, [0,0,0,0], t3,args=(hnu0,t0+t3[0]-t2[0],sigma0,tau1,tau2,esat,gamma,alpha),hmax=1.0)
	exp=np.exp(((t2)/tau3))*np.heaviside((-t2+t0), 0)/tau3
	#ps.plotkinetictrace(exp,t2,save=1,destination="testhevix.txt",logx=0)
	#ps.plotkinetictrace(s[:,3],t2,save=1,destination="testhevix.txt",logx=0)
	#print("xxxxxxx")
	#print(len(exp))
	#print(type(exp))
	#print(np.shape(s[:,3]))
	#print(type(s))
	ssexp=np.convolve(s[:,3], exp, mode="full")
	s2=scalar*ssexp[9999:29999]
	#print(np.shape(ssexp[9999:29999]))
	#ps.plotkinetictrace(ssexp[9999:29999],t2,save=1,destination="testhevix.txt",logx=0)
	#print("xxxxxxx")
	idx=[(np.abs(t2 - tx)).argmin() for tx in t]
	pm=[np.sign(t2[idx[j]]) for j in range(len(idx))]
	#print("idx:",idx)
	#print("t",t)
	#print("t2[idx]",t2[idx])
	#print("pm:",pm)
	s3=[]
	#linearly interpolate ssexp between interval of t2
	for i in range(len(idx)):
		if pm[i]==0:
			s3.append(s2[idx[i]])
		elif pm[i]<0:
			s3.append(s2[idx[i]-1])
		elif pm[i]>0:
			s3.append(s2[idx[i]])
	return s3

def esat_expconv_global(t,gamma,alpha):
	hnu0
	scalar
	t0
	sigma0
	tau1,tau2,tau3,esat
	if t[-1]<t[0]:
		print("XXXXX ERRROR XXXXX time array must be increasing odeint to work.")
	#buffer the time domain to be twice as big so we don't get edge effects.
	t2=np.linspace(-10000,9999,20000)
	t3=t2-t2[0]
	#idt0=(np.abs(t2 - t0)).argmin()
	s =odeint(e_gen_sat6, [0,0,0,0], t3,args=(hnu0,t0+t3[0]-t2[0],sigma0,tau1,tau2,esat,gamma,alpha),hmax=1.0)
	exp=np.exp(((t2)/tau3))*np.heaviside((-t2+t0), 0)/tau3
	#ps.plotkinetictrace(exp,t2,save=1,destination="testhevix.txt",logx=0)
	#ps.plotkinetictrace(s[:,3],t2,save=1,destination="testhevix.txt",logx=0)
	#print("xxxxxxx")
	#print(len(exp))
	#print(type(exp))
	#print(np.shape(s[:,3]))
	#print(type(s))
	ssexp=np.convolve(s[:,3], exp, mode="full")
	s2=scalar*ssexp[9999:29999]
	#print(np.shape(ssexp[9999:29999]))
	#ps.plotkinetictrace(ssexp[9999:29999],t2,save=1,destination="testhevix.txt",logx=0)
	#print("xxxxxxx")
	idx=[(np.abs(t2 - tx)).argmin() for tx in t]
	pm=[np.sign(t2[idx[j]]) for j in range(len(idx))]
	#print("idx:",idx)
	#print("t",t)
	#print("t2[idx]",t2[idx])
	#print("pm:",pm)
	s3=[]
	#linearly interpolate ssexp between interval of t2
	for i in range(len(idx)):
		if pm[i]==0:
			s3.append(s2[idx[i]])
		elif pm[i]<0:
			s3.append(s2[idx[i]-1])
		elif pm[i]>0:
			s3.append(s2[idx[i]])
	return s3

def testglobaltimeseries(traces,timepoints,p0,hold,bounds,alpha=.05,special=[],func='esat_expconv',verbose=0):
	global ghold
	ghold=hold
	global gp0
	gp0=p0
	global gtraces
	gtraces=traces
	global gtimepoints
	gtimepoints=timepoints
	global gspecial
	gspecial=special
	print("Special Paramter:",gspecial)
	pfree=np.array([])
	lowerbound=[]
	upperbound=[]
	for k in range(len(hold)):
		if not hold[k]:
			pfree=np.concatenate([pfree,[p0[k]]])
			lowerbound.append(bounds[0][k])
			upperbound.append(bounds[1][k])
	boundsfree=(lowerbound,upperbound)
	testdata=esat_expconv_global(timepoints,*p0)
	return testdata

def esat_expconv_global(ta,*pa):
	p=np.array([])
	counter=0
	for h in range(len(ghold)):
		if ghold[h]:
			p=np.concatenate([p,[gp0[h]]])
		else:
			p=np.concatenate([p,[pa[counter]]])
			counter+=1
	t=gtimepoints
	if t[-1]<t[0]:
		print("XXXXX ERRROR XXXXX time array must be increasing odeint to work.")
	#buffer the time domain to be twice as big so we don't get edge effects.
	t2=np.linspace(-10000,9999,20000)
	t3=t2-t2[0]
	#idt0=(np.abs(t2 - t0)).argmin()
	#s =odeint(eh_gen_sat, [0,0,0,0,0,0], t3,args=(hnu0,t0+t3[0]-t2[0],sigma0,tau1,esat,tau2,hsat),hmax=1.0)
	globalfit=np.array([])
	for g in range(gtraces):
		s =odeint(e_gen_sat6, [0,0,0,0], t3,args=(gspecial[g],p[1]+t3[0]-t2[0],p[2],p[3],p[4],p[5],p[7],p[8]),hmax=1.0)
		exp=np.exp(((t2)/p[6]))*np.heaviside((-t2+p[3]), 0)/p[6]
		ssexp=np.convolve(s[:,3], exp, mode="full")
		s2=scalar*ssexp[9999:29999]
		idx=[(np.abs(t2 - tx)).argmin() for tx in t]
		pm=[np.sign(t2[idx[j]]) for j in range(len(idx))]
		s3=[]
	#linearly interpolate ssexp between interval of t2
		for i in range(len(idx)):
			if pm[i]==0:
				s3.append(s2[idx[i]])
			elif pm[i]<0:
				s3.append(s2[idx[i]-1])
			elif pm[i]>0:
				s3.append(s2[idx[i]])
		globalfit=np.concatenate([globalfit,s3])
	return globalfit


def fitglobaltimeseries(filelist,timepoints,p0,hold,bounds,alpha=.05,special=[],func='eh_gen_sat_interp_attenuation_global',verbose=0):
	global ghold
	ghold=hold
	global gp0
	gp0=p0
	global gtraces
	gtraces=len(filelist)
	global gtimepoints
	gtimepoints=timepoints
	global gspecial
	gspecial=special
	print("Special Paramter:",gspecial)
	pfree=np.array([])
	lowerbound=[]
	upperbound=[]
	for k in range(len(hold)):
		if not hold[k]:
			pfree=np.concatenate([pfree,[p0[k]]])
			lowerbound.append(bounds[0][k])
			upperbound.append(bounds[1][k])
	boundsfree=(lowerbound,upperbound)
	#optparamsall=np.array([], dtype=float).reshape(0,len(p0))
	#uncertaintiesall=np.array([], dtype=float).reshape(0,len(p0))
	#fitdataall=np.array([], dtype=float).reshape(0,len(timepoints))
	#calculate student T for confidnece interval
	n = len(timepoints)    # number of data points
	p = len(p0) # number of parameters
	dof = max(0, n - p) # number of degrees of freedom
	tval = t.ppf(1.0-alpha/2., dof) # student-t value for the dof and confidence level
	#bounds=ps.createbounds(p0,boundfactor=1.1)
	datamulti=np.array([])
	timepointsmulti=np.array([])
	for f in filelist:
		data =np.loadtxt(f)
		datamulti=np.concatenate([datamulti,data])
		timepointsmulti=np.concatenate([timepointsmulti,timepoints])
	if verbose:
		print("Prosessing :", f)
		start_time = time.time()
		print("initial guess: ", p0)
		print("lower bounds: ", bounds[0])
		print("upper bounds: ", bounds[1])
	if np.sum(hold)==len(hold):
		print("All parameters constrained")
		popt=np.array([])
		pcov=np.array([])
		fitdata=eh_gen_sat_interp_attenuation_global(timepoints,*p0)
	else:
		popt, pcov = opt.curve_fit(eh_gen_sat_interp_attenuation_global,timepointsmulti, datamulti, p0=pfree,bounds=boundsfree,verbose=1)
		fitdata=eh_gen_sat_interp_attenuation_global(timepoints,*popt)
		if verbose:
			print("--- %s (s) to fit kinetic trace --- " % (time.time() - start_time))
			print("Optimization Parameters: ",popt)
		#optparamsall=np.vstack((optparamsall,popt))
		uncertaintiesall = np.multiply(tval,np.sqrt(np.diagonal(pcov)))
		fnopt=getfileprefix(filelist[0],delimiter="_")+"_all_popt.txt"
		fnvar=getfileprefix(filelist[0],delimiter="_")+"_all_pvar.txt"
		np.savetxt(fnvar,uncertaintiesall)
		np.savetxt(fnopt,popt)
	fitdataall=np.reshape(fitdata,(len(filelist),int(np.shape(fitdata)[0]/len(filelist))))
	counter=0
	for f in filelist:
		fn=generatefilename(f,suffix="_fit",extension="txt",verbose=verbose)
		np.savetxt(fn,fitdataall[counter,:])
		counter+=1
	fnfitall=getfileprefix(filelist[0],delimiter="_")+"_all_fit.txt"
	np.savetxt(fnfitall,fitdataall)
	return

def esat_expgaussdoubleconv_hold(t,a,b,t0,tau2,gamma):
	hnu0=0.56
	sigma0=14
	tau3=65
	esat=0.2
	tau1=14.0
	if t[-1]<t[0]:
		print("XXXXX ERRROR XXXXX time array must be increasing odeint to work.")
	#buffer the time domain to be twice as big so we don't get edge effects.
	t2=np.linspace(-20000,19999,20000)
	t3=t2-t2[0]
	#idt0=(np.abs(t2 - t0)).argmin()
	s =odeint(e_gen_sat7, [0,0,0,0], t3,args=(hnu0,t0+t3[0]-t2[0],sigma0,tau1,tau2,esat,gamma),hmax=1.0)
	exp=np.exp(((t2)/tau3))*np.heaviside((-t2+t0), 0)/tau3
	gauss=np.exp(-((t2-t0)/sigma0)**2/2)/(2*3.14152*sigma0)
	#ps.plotkinetictrace(exp,t2,save=1,destination="testhevix.txt",logx=0)
	#ps.plotkinetictrace(s[:,3],t2,save=1,destination="testhevix.txt",logx=0)
	#print("xxxxxxx")
	#print(len(exp))
	#print(type(exp))
	#print(np.shape(s[:,3]))
	#print(type(s))
	ssexp=np.convolve(s[:,3], exp, mode="full")
	ssgauss=np.convolve(s[:,3], gauss, mode="full")
	s2=a*ssgauss[9999:29999]+b*ssexp[9999:29999]
	#print(np.shape(ssexp[9999:29999]))
	#ps.plotkinetictrace(ssexp[9999:29999],t2,save=1,destination="testhevix.txt",logx=0)
	#print("xxxxxxx")
	idx=[(np.abs(t2 - tx)).argmin() for tx in t]
	pm=[np.sign(t2[idx[j]]) for j in range(len(idx))]
	#print("idx:",idx)
	#print("t",t)
	#print("t2[idx]",t2[idx])
	#print("pm:",pm)
	s3=[]
	#linearly interpolate ssexp between interval of t2
	for i in range(len(idx)):
		if pm[i]==0:
			s3.append(s2[idx[i]])
		elif pm[i]<0:
			s3.append(s2[idx[i]-1])
		elif pm[i]>0:
			s3.append(s2[idx[i]])
	return s3

def esat_expgaussdoubleconv_hold2(t,hnu0,a,b,t0,tau2,gamma):
	sigma0=14
	tau3=65
	esat=0.2
	tau1=14.0
	if t[-1]<t[0]:
		print("XXXXX ERRROR XXXXX time array must be increasing odeint to work.")
	#buffer the time domain to be twice as big so we don't get edge effects.
	t2=np.linspace(-20000,19999,20000)
	t3=t2-t2[0]
	s =odeint(e_gen_sat7, [0,0,0,0], t3,args=(hnu0,t0+t3[0]-t2[0],sigma0,tau1,tau2,esat,gamma),hmax=1.0)
	exp=np.exp(((t2)/tau3))*np.heaviside((-t2+t0), 0)/tau3
	gauss=np.exp(-((t2-t0)/sigma0)**2/2)/(2*3.14152*sigma0)
	ssexp=np.convolve(s[:,3], exp, mode="full")
	ssgauss=np.convolve(s[:,3], gauss, mode="full")
	s2=a*ssgauss[9999:29999]+b*ssexp[9999:29999]
	idx=[(np.abs(t2 - tx)).argmin() for tx in t]
	pm=[np.sign(t2[idx[j]]) for j in range(len(idx))]
	s3=[]
	#linearly interpolate ssexp between interval of t2
	for i in range(len(idx)):
		if pm[i]==0:
			s3.append(s2[idx[i]])
		elif pm[i]<0:
			s3.append(s2[idx[i]-1])
		elif pm[i]>0:
			s3.append(s2[idx[i]])
	return s3

def esat_expgaussdoubleconv_global(ta,*pa):
	p=np.array([])
	counter=0
	for h in range(len(ghold)):
		if ghold[h]:
			p=np.concatenate([p,[gp0[h]]])
		else:
			p=np.concatenate([p,[pa[counter]]])
			counter+=1
	t=gtimepoints
	if t[-1]<t[0]:
		print("XXXXX ERRROR XXXXX time array must be increasing odeint to work.")
	#buffer the time domain to be twice as big so we don't get edge effects.
	t2=np.linspace(-20000,19999,20000)
	t3=t2-t2[0]
	#idt0=(np.abs(t2 - t0)).argmin()
	#s =odeint(eh_gen_sat, [0,0,0,0,0,0], t3,args=(hnu0,t0+t3[0]-t2[0],sigma0,tau1,esat,tau2,hsat),hmax=1.0)
	globalfit=np.array([])
	for g in range(gtraces):
		s =odeint(e_gen_sat7, [0,0,0,0], t3,args=(hnu0,t0+t3[0]-t2[0],sigma0,tau1,tau2,esat,gamma),hmax=1.0)
		exp=np.exp(((t2)/tau3))*np.heaviside((-t2+t0), 0)/tau3
		gauss=np.exp(-((t2-t0)/sigma0)**2/2)/(2*3.14152*sigma0)
		ssexp=np.convolve(s[:,3], exp, mode="full")
		ssgauss=np.convolve(s[:,3], gauss, mode="full")
		s2=a*ssgauss[9999:29999]+b*ssexp[9999:29999]
		idx=[(np.abs(t2 - tx)).argmin() for tx in t]
		pm=[np.sign(t2[idx[j]]) for j in range(len(idx))]
		s3=[]
	#linearly interpolate ssexp between interval of t2
		for i in range(len(idx)):
			if pm[i]==0:
				s3.append(s2[idx[i]])
			elif pm[i]<0:
				s3.append(s2[idx[i]-1])
			elif pm[i]>0:
				s3.append(s2[idx[i]])
		globalfit=np.concatenate([globalfit,pvinterp])
	return globalfit

def e_gen_sat_cw(y,t,hnu0,t0,sigma0,tau1,tau2,esat,gamma,alpha):
	"""
	t=np.linspace(0,7000,20000)
	powerseries= np.logspace(-1,4, num=11,base=np.e)
	fig, ax = plt.subplots()
	colors = [plt.cm.jet(i) for i in np.linspace(0, 1,len(powerseries))]
	ax.set_prop_cycle('color', colors)
	for i in powerseries:
		label0=str(i)[0:4]
		#hnu0,t0,sigma0,tau1,esat,tau2,hsat
		b =odeint(ps.eh_gen_sat, [0,0,0,0,0,0], t,args=(i,1000.,100.,350,1,10000,.5))
		plt.plot(t,b[:,3]-b[:,5],label=label0)
	plt.rcParams.update({'font.size': 10})  # increase the font size
	plt.xlabel("time Delay (ps")
	plt.ylabel("Photovoltage [e_\{dr\}]-[h_\{dr\}]")
	plt.legend(loc='best')
	#plt.xlim(-202,-198)
	#plt.xticks([10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000]) 
	plt.show()
	"""
	#note t must start at 0
	t2=y[0]
	g=y[1]
	eb=y[2]
	edr=y[3]
	#dg=0
	dt=t-t2
	#hnu0 is microjoules/cm^2/ps
	#1 uj/cm^2/ps =  2.59947e+24 photons/cm^2/s
	#for beam diameter of 30 um this is 28 watts
	#note t interval must be spaced by 1 ps
	dg=hnu0*(np.heaviside(t2-t0, 0))
	deb=dg-1/tau1*(1-edr/esat)*eb
	dedr=1/tau1*(1-edr/esat)*eb-1/tau2*edr*(1+gamma*edr)**alpha
	return [dt,dg,deb,dedr]

def negtimeconv(t,hnu0,scalar,t0,sigma0,tau1,tau2,tau3,esat,gamma,alpha):
	if t[-1]<t[0]:
		print("XXXXX ERRROR XXXXX time array must be increasing odeint to work.")
	#buffer the time domain to be twice as big so we don't get edge effects.
	t2=np.linspace(-10000,9999,2000)
	t3=t2-t2[0]
	#idt0=(np.abs(t2 - t0)).argmin()
	s =odeint(e_gen_sat3, [0,0,0,0], t3,args=(hnu0,t0+t3[0]-t2[0],sigma0,tau1,tau2,esat,gamma,alpha),hmax=1.0)
	exp=np.exp(((t2)/tau3))*np.heaviside((-t2+t0), 0)
	#ps.plotkinetictrace(exp,t2,save=1,destination="testhevix.txt",logx=0)
	#ps.plotkinetictrace(s[:,3],t2,save=1,destination="testhevix.txt",logx=0)
	#print("xxxxxxx")
	#print(len(exp))
	#print(type(exp))
	#print(np.shape(s[:,3]))
	#print(type(s))
	ssexp=np.convolve(s[:,3], exp, mode="full")
	s2=ssexp[999:2999]
	#print(np.shape(ssexp[9999:29999]))
	#ps.plotkinetictrace(ssexp[9999:29999],t2,save=1,destination="testhevix.txt",logx=0)
	#print("xxxxxxx")
	idx=[(np.abs(t2 - tx)).argmin() for tx in t]
	pm=[np.sign(t2[idx[j]]) for j in range(len(idx))]
	#print("idx:",idx)
	#print("t",t)
	#print("t2[idx]",t2[idx])
	#print("pm:",pm)
	s3=[]
	#linearly interpolate ssexp between interval of t2
	for i in range(len(idx)):
		if pm[i]==0:
			s3.append(s2[idx[i]])
		elif pm[i]<0:
			s3.append(s2[idx[i]-1])
		elif pm[i]>0:
			s3.append(s2[idx[i]])
	return s3

def negtimeconv_cw(t,hnu0,scalar,t0,sigma0,tau1,tau2,tau3,esat,gamma,alpha):
	if t[-1]<t[0]:
		print("XXXXX ERRROR XXXXX time array must be increasing odeint to work.")
	numpoints= 90000
	#buffer the time domain to be twice as big so we don't get edge effects.
	t2=np.linspace(-10000,9999,20000)
	t3=t2-t2[0]
	#idt0=(np.abs(t2 - t0)).argmin()
	s =odeint(e_gen_sat_cw, [0,0,0,0], t3,args=(hnu0,t0+t3[0]-t2[0],sigma0,tau1,tau2,esat,gamma,alpha),hmax=1.0)
	exp=np.exp(((t2)/tau3))*np.heaviside((-t2+t0), 0)
	#ps.plotkinetictrace(exp,t2,save=1,destination="testhevix.txt",logx=0)
	#ps.plotkinetictrace(s[:,3],t2,save=1,destination="testhevix.txt",logx=0)
	#print("xxxxxxx")
	#print(len(exp))
	#print(type(exp))
	#print(np.shape(s[:,3]))
	#print(type(s))
	ssexp=np.convolve(s[:,3], exp, mode="full")
	#plotkinetictrace(s[:,1],t2,save=1,destination="testhevix.txt",logx=0)
	s2=ssexp[9999:29999]*2/47.1
	#print(np.shape(ssexp[9999:29999]))
	#ps.plotkinetictrace(ssexp[9999:29999],t2,save=1,destination="testhevix.txt",logx=0)
	#print("xxxxxxx")
	idx=[(np.abs(t2 - tx)).argmin() for tx in t]
	pm=[np.sign(t2[idx[j]]) for j in range(len(idx))]
	#print("idx:",idx)
	#print("t",t)
	#print("t2[idx]",t2[idx])
	#print("pm:",pm)
	s3=[]
	#linearly interpolate ssexp between interval of t2
	for i in range(len(idx)):
		if pm[i]==0:
			s3.append(s2[idx[i]])
		elif pm[i]<0:
			s3.append(s2[idx[i]-1])
		elif pm[i]>0:
			s3.append(s2[idx[i]])
	return s3

def negtimeconv_cw_lin(t,hnu0,scalar,t0,sigma0,tau1,tau2,tau3,esat,gamma,alpha):
	t2=t-t0
	#idt0=(np.abs(t2 - t0)).argmin()
	s =odeint(e_gen_sat_cw, [0,0,0,0], t2,args=(hnu0,t0,sigma0,tau1,tau2,esat,gamma,alpha),hmax=1.0)
	plotkinetictrace(s[:,1],t2,save=1,destination="testhevix.txt",logx=0)
	return s[:,3]

def e_gen_sat_interp3(t,hnu0,scalar,t0,sigma0,tau1,tau2,esat,gamma,alpha):
	"""
	p0=np.array([.001,12.92,-25,9.2,350,.0068,11000.1,.004,180])
	t=np.linspace(-1000,8000,20000)
	powerseries= np.logspace(-3,3, num=10,base=np.e)
	t0=0
	sigma0=10
	tau1=350
	esat=1
	tau2=10000
	hsat=.5
	fig, ax = plt.subplots()
	colors = [plt.cm.jet(i) for i in np.linspace(0, 1,len(powerseries))]
	ax.set_prop_cycle('color', colors)
	for i in powerseries:
		label0=str(i)[0:4]
		b =ps.eh_gen_sat_interp(t,i,t0,sigma0,tau1,esat,tau2,hsat)
		plt.plot(t,b,label=label0)
	ax.set_xlabel("time Delay (ps)")
	ax.set_ylabel(r'Photovoltage $[e_{dr}]-[h_{dr}]$')
	plt.legend(loc='best')
	plt.show()
	"""
	if t[-1]<t[0]:
		print("XXXXX ERRROR XXXXX time array must be increasing odeint to work.")
	numpoints= 90000
	#buffer the time domain to be twice as big so we don't get edge effects.
	t2=np.linspace(-6000,t[-1],20000)
	t3=t2-t2[0]
	#idt0=(np.abs(t2 - t0)).argmin()
	s =odeint(e_gen_sat3, [0,0,0,0], t3,args=(hnu0,t0+t3[0]-t2[0],sigma0,tau1,tau2,esat,gamma,alpha),hmax=1.0)
	pv=s[:,3]*scalar
	idx=[(np.abs(t2 - tx)).argmin() for tx in t]
	pm=[np.sign(t2[idx[j]]-t[j]) for j in range(len(idx))]
	pvinterp=np.array([])
	#pvinterp_at=np.array([])
	#linearly interpolate ssgauss between interval of t2
	for i in range(len(idx)):
		if pm[i]==0:
			pvinterp=np.concatenate([pvinterp,[pv[idx[i]]]])
			#pvinterp_at=np.concatenate([pvinterp_at,[pv_at[idx[i]]]])
		elif pm[i]<0:
			pvinterp=np.concatenate([pvinterp,[pv[idx[i]-1]+(pv[idx[i]]-pv[idx[i]-1])*(t[i]-t2[idx[i]-1])/(t2[idx[i]]-t2[idx[i]-1])]])
			#pvinterp_at=np.concatenate([pvinterp_at,[pv_at[idx[i]-1]+(pv_at[idx[i]]-pv_at[idx[i]-1])*(t[i]-t2[idx[i]-1])/(t2[idx[i]]-t2[idx[i]-1])]])
		elif pm[i]>0:
			pvinterp=np.concatenate([pvinterp,[pv[idx[i]]+(pv[idx[i]+1]-pv[idx[i]])*(t[i]-t2[idx[i]])/(t2[idx[i]+1]-t2[idx[i]])]])
			#pvinterp_at=np.concatenate([pvinterp_at,[pv_at[idx[i]]+(pv_at[idx[i]+1]-pv_at[idx[i]])*(t[i]-t2[idx[i]])/(t2[idx[i]+1]-t2[idx[i]])]])
	return pvinterp

def e_gen_sat_interp2(t,hnu0,scalar,t0,sigma0,tau1,tau2,esat):
	"""
	p0=np.array([.001,12.92,-25,9.2,350,.0068,11000.1,.004,180])
	t=np.linspace(-1000,8000,20000)
	powerseries= np.logspace(-3,3, num=10,base=np.e)
	t0=0
	sigma0=10
	tau1=350
	esat=1
	tau2=10000
	hsat=.5
	fig, ax = plt.subplots()
	colors = [plt.cm.jet(i) for i in np.linspace(0, 1,len(powerseries))]
	ax.set_prop_cycle('color', colors)
	for i in powerseries:
		label0=str(i)[0:4]
		b =ps.eh_gen_sat_interp(t,i,t0,sigma0,tau1,esat,tau2,hsat)
		plt.plot(t,b,label=label0)
	ax.set_xlabel("time Delay (ps)")
	ax.set_ylabel(r'Photovoltage $[e_{dr}]-[h_{dr}]$')
	plt.legend(loc='best')
	plt.show()
	"""
	if t[-1]<t[0]:
		print("XXXXX ERRROR XXXXX time array must be increasing odeint to work.")
	numpoints= 90000
	#buffer the time domain to be twice as big so we don't get edge effects.
	t2=np.linspace(-6000,t[-1],20000)
	t3=t2-t2[0]
	#idt0=(np.abs(t2 - t0)).argmin()
	s =odeint(e_gen_sat2, [0,0,0,0], t3,args=(hnu0,t0+t3[0]-t2[0],sigma0,tau1,tau2,esat),hmax=1.0)
	pv=s[:,3]*scalar
	idx=[(np.abs(t2 - tx)).argmin() for tx in t]
	pm=[np.sign(t2[idx[j]]-t[j]) for j in range(len(idx))]
	pvinterp=np.array([])
	#pvinterp_at=np.array([])
	#linearly interpolate ssgauss between interval of t2
	for i in range(len(idx)):
		if pm[i]==0:
			pvinterp=np.concatenate([pvinterp,[pv[idx[i]]]])
			#pvinterp_at=np.concatenate([pvinterp_at,[pv_at[idx[i]]]])
		elif pm[i]<0:
			pvinterp=np.concatenate([pvinterp,[pv[idx[i]-1]+(pv[idx[i]]-pv[idx[i]-1])*(t[i]-t2[idx[i]-1])/(t2[idx[i]]-t2[idx[i]-1])]])
			#pvinterp_at=np.concatenate([pvinterp_at,[pv_at[idx[i]-1]+(pv_at[idx[i]]-pv_at[idx[i]-1])*(t[i]-t2[idx[i]-1])/(t2[idx[i]]-t2[idx[i]-1])]])
		elif pm[i]>0:
			pvinterp=np.concatenate([pvinterp,[pv[idx[i]]+(pv[idx[i]+1]-pv[idx[i]])*(t[i]-t2[idx[i]])/(t2[idx[i]+1]-t2[idx[i]])]])
			#pvinterp_at=np.concatenate([pvinterp_at,[pv_at[idx[i]]+(pv_at[idx[i]+1]-pv_at[idx[i]])*(t[i]-t2[idx[i]])/(t2[idx[i]+1]-t2[idx[i]])]])
	return pvinterp

def eh_gen_sat_interp(t,hnu0,t0,sigma0,tau1,esat,tau2,hsat):
	"""
	t=np.linspace(-1000,8000,20000)
	powerseries= np.logspace(-3,3, num=10,base=np.e)
	t0=0
	sigma0=10
	tau1=350
	esat=1
	tau2=10000
	hsat=.5
	fig, ax = plt.subplots()
	colors = [plt.cm.jet(i) for i in np.linspace(0, 1,len(powerseries))]
	ax.set_prop_cycle('color', colors)
	for i in powerseries:
		label0=str(i)[0:4]
		b =ps.eh_gen_sat_interp(t,i,t0,sigma0,tau1,esat,tau2,hsat)
		plt.plot(t,b,label=label0)
	ax.set_xlabel("time Delay (ps)")
	ax.set_ylabel(r'Photovoltage $[e_{dr}]-[h_{dr}]$')
	plt.legend(loc='best')
	plt.show()
	"""
	if t[-1]<t[0]:
		print("XXXXX ERRROR XXXXX time array must be increasing odeint to work.")
	numpoints= 90000
	#buffer the time domain to be twice as big so we don't get edge effects.
	t2=np.linspace(-6000,t[-1],20000)
	t3=t2-t2[0]
	#idt0=(np.abs(t2 - t0)).argmin()
	s =odeint(eh_gen_sat, [0,0,0,0,0,0], t3,args=(hnu0,t0+t3[0]-t2[0],sigma0,tau1,esat,tau2,hsat),hmax=1.0)
	pv=s[:,3]-s[:,5]
	idx=[(np.abs(t2 - tx)).argmin() for tx in t]
	pm=[np.sign(t2[idx[j]]-t[j]) for j in range(len(idx))]
	pvinterp=[]
	#linearly interpolate ssgauss between interval of t2
	for i in range(len(idx)):
		if pm[i]==0:
			pvinterp.append(pv[idx[i]])
		elif pm[i]<0:
			pvinterp.append(pv[idx[i]-1]+(pv[idx[i]]-pv[idx[i]-1])*(t[i]-t2[idx[i]-1])/(t2[idx[i]]-t2[idx[i]-1]))
		elif pm[i]>0:
			pvinterp.append(pv[idx[i]]+(pv[idx[i]+1]-pv[idx[i]])*(t[i]-t2[idx[i]])/(t2[idx[i]+1]-t2[idx[i]]))
	return pvinterp

def eh_gen_sat_interp_attenuation(t,hnu0,scalar,t0,sigma0,tau1,esat,tau2,hsat,ar):
	"""
	p0=np.array([.001,12.92,-25,9.2,350,.0068,11000.1,.004,180])
	t=np.linspace(-1000,8000,20000)
	powerseries= np.logspace(-3,3, num=10,base=np.e)
	t0=0
	sigma0=10
	tau1=350
	esat=1
	tau2=10000
	hsat=.5
	fig, ax = plt.subplots()
	colors = [plt.cm.jet(i) for i in np.linspace(0, 1,len(powerseries))]
	ax.set_prop_cycle('color', colors)
	for i in powerseries:
		label0=str(i)[0:4]
		b =ps.eh_gen_sat_interp(t,i,t0,sigma0,tau1,esat,tau2,hsat)
		plt.plot(t,b,label=label0)
	ax.set_xlabel("time Delay (ps)")
	ax.set_ylabel(r'Photovoltage $[e_{dr}]-[h_{dr}]$')
	plt.legend(loc='best')
	plt.show()
	"""
	if t[-1]<t[0]:
		print("XXXXX ERRROR XXXXX time array must be increasing odeint to work.")
	numpoints= 90000
	#buffer the time domain to be twice as big so we don't get edge effects.
	t2=np.linspace(-6000,t[-1],20000)
	t3=t2-t2[0]
	#idt0=(np.abs(t2 - t0)).argmin()
	s =odeint(eh_gen_sat, [0,0,0,0,0,0], t3,args=(hnu0,t0+t3[0]-t2[0],sigma0,tau1,esat,tau2,hsat),hmax=1.0)
	s_at=odeint(eh_gen_sat, [0,0,0,0,0,0], t3,args=(hnu0/ar,t0+t3[0]-t2[0],sigma0,tau1,esat,tau2,hsat),hmax=1.0)
	pv=s[:,3]-s[:,5]
	pv_at=s_at[:,3]-s_at[:,5]
	pv2=(pv-pv_at)*scalar
	idx=[(np.abs(t2 - tx)).argmin() for tx in t]
	pm=[np.sign(t2[idx[j]]-t[j]) for j in range(len(idx))]
	pvinterp=np.array([])
	#pvinterp_at=np.array([])
	#linearly interpolate ssgauss between interval of t2
	for i in range(len(idx)):
		if pm[i]==0:
			pvinterp=np.concatenate([pvinterp,[pv2[idx[i]]]])
			#pvinterp_at=np.concatenate([pvinterp_at,[pv_at[idx[i]]]])
		elif pm[i]<0:
			pvinterp=np.concatenate([pvinterp,[pv2[idx[i]-1]+(pv2[idx[i]]-pv2[idx[i]-1])*(t[i]-t2[idx[i]-1])/(t2[idx[i]]-t2[idx[i]-1])]])
			#pvinterp_at=np.concatenate([pvinterp_at,[pv_at[idx[i]-1]+(pv_at[idx[i]]-pv_at[idx[i]-1])*(t[i]-t2[idx[i]-1])/(t2[idx[i]]-t2[idx[i]-1])]])
		elif pm[i]>0:
			pvinterp=np.concatenate([pvinterp,[pv2[idx[i]]+(pv2[idx[i]+1]-pv2[idx[i]])*(t[i]-t2[idx[i]])/(t2[idx[i]+1]-t2[idx[i]])]])
			#pvinterp_at=np.concatenate([pvinterp_at,[pv_at[idx[i]]+(pv_at[idx[i]+1]-pv_at[idx[i]])*(t[i]-t2[idx[i]])/(t2[idx[i]+1]-t2[idx[i]])]])
	return pvinterp

def e_gen_sat_interp(t,hnu0,scalar,t0,sigma0,tau1,esat):
	"""
	p0=np.array([.001,12.92,-25,9.2,350,.0068,11000.1,.004,180])
	t=np.linspace(-1000,8000,20000)
	powerseries= np.logspace(-3,3, num=10,base=np.e)
	t0=0
	sigma0=10
	tau1=350
	esat=1
	tau2=10000
	hsat=.5
	fig, ax = plt.subplots()
	colors = [plt.cm.jet(i) for i in np.linspace(0, 1,len(powerseries))]
	ax.set_prop_cycle('color', colors)
	for i in powerseries:
		label0=str(i)[0:4]
		b =ps.eh_gen_sat_interp(t,i,t0,sigma0,tau1,esat,tau2,hsat)
		plt.plot(t,b,label=label0)
	ax.set_xlabel("time Delay (ps)")
	ax.set_ylabel(r'Photovoltage $[e_{dr}]-[h_{dr}]$')
	plt.legend(loc='best')
	plt.show()
	"""
	if t[-1]<t[0]:
		print("XXXXX ERRROR XXXXX time array must be increasing odeint to work.")
	numpoints= 90000
	#buffer the time domain to be twice as big so we don't get edge effects.
	t2=np.linspace(-6000,t[-1],20000)
	t3=t2-t2[0]
	#idt0=(np.abs(t2 - t0)).argmin()
	s =odeint(e_gen_sat, [0,0,0,0], t3,args=(hnu0,t0+t3[0]-t2[0],sigma0,tau1,esat),hmax=1.0)
	pv=s[:,3]*scalar
	idx=[(np.abs(t2 - tx)).argmin() for tx in t]
	pm=[np.sign(t2[idx[j]]-t[j]) for j in range(len(idx))]
	pvinterp=np.array([])
	#pvinterp_at=np.array([])
	#linearly interpolate ssgauss between interval of t2
	for i in range(len(idx)):
		if pm[i]==0:
			pvinterp=np.concatenate([pvinterp,[pv[idx[i]]]])
			#pvinterp_at=np.concatenate([pvinterp_at,[pv_at[idx[i]]]])
		elif pm[i]<0:
			pvinterp=np.concatenate([pvinterp,[pv[idx[i]-1]+(pv[idx[i]]-pv[idx[i]-1])*(t[i]-t2[idx[i]-1])/(t2[idx[i]]-t2[idx[i]-1])]])
			#pvinterp_at=np.concatenate([pvinterp_at,[pv_at[idx[i]-1]+(pv_at[idx[i]]-pv_at[idx[i]-1])*(t[i]-t2[idx[i]-1])/(t2[idx[i]]-t2[idx[i]-1])]])
		elif pm[i]>0:
			pvinterp=np.concatenate([pvinterp,[pv[idx[i]]+(pv[idx[i]+1]-pv[idx[i]])*(t[i]-t2[idx[i]])/(t2[idx[i]+1]-t2[idx[i]])]])
			#pvinterp_at=np.concatenate([pvinterp_at,[pv_at[idx[i]]+(pv_at[idx[i]+1]-pv_at[idx[i]])*(t[i]-t2[idx[i]])/(t2[idx[i]+1]-t2[idx[i]])]])
	return pvinterp

def eh_gen_sat_interp_attenuation_global(ta,*pa):
	"""
	t=np.linspace(-1000,8000,20000)
	powerseries= np.logspace(-3,3, num=10,base=np.e)
	t0=0
	sigma0=10
	tau1=350
	esat=1
	tau2=10000
	hsat=.5
	fig, ax = plt.subplots()
	colors = [plt.cm.jet(i) for i in np.linspace(0, 1,len(powerseries))]
	ax.set_prop_cycle('color', colors)
	for i in powerseries:
		label0=str(i)[0:4]
		b =ps.eh_gen_sat_interp(t,i,t0,sigma0,tau1,esat,tau2,hsat)
		plt.plot(t,b,label=label0)
	ax.set_xlabel("time Delay (ps)")
	ax.set_ylabel(r'Photovoltage $[e_{dr}]-[h_{dr}]$')
	plt.legend(loc='best')
	plt.show()
	"""
	p=np.array([])
	counter=0
	for h in range(len(ghold)):
		if ghold[h]:
			p=np.concatenate([p,[gp0[h]]])
		else:
			p=np.concatenate([p,[pa[counter]]])
			counter+=1
	t=gtimepoints
	if t[-1]<t[0]:
		print("XXXXX ERRROR XXXXX time array must be increasing odeint to work.")
	numpoints= 90000
	#buffer the time domain to be twice as big so we don't get edge effects.
	t2=np.linspace(-6000,t[-1],20000)
	t3=t2-t2[0]
	#idt0=(np.abs(t2 - t0)).argmin()
	#s =odeint(eh_gen_sat, [0,0,0,0,0,0], t3,args=(hnu0,t0+t3[0]-t2[0],sigma0,tau1,esat,tau2,hsat),hmax=1.0)
	globalfit=np.array([])
	for g in range(gtraces):
		s =odeint(eh_gen_sat, [0,0,0,0,0,0], t3,args=(gspecial[g],p[1]+t3[0]-t2[0],p[2],p[3],p[4],p[5],p[6]),hmax=1.0)
		s_at=odeint(eh_gen_sat, [0,0,0,0,0,0], t3,args=(gspecial[g]/p[7],p[1]+t3[0]-t2[0],p[2],p[3],p[4],p[5],p[6]),hmax=1.0)
		pv=s[:,3]-s[:,5]
		pv_at=s_at[:,3]-s_at[:,5]
		pv2=p[0]*(pv-pv_at)
		idx=[(np.abs(t2 - tx)).argmin() for tx in t]
		pm=[np.sign(t2[idx[j]]-t[j]) for j in range(len(idx))]
		pvinterp=np.array([])
		#pvinterp_at=np.array([])
		#linearly interpolate ssgauss between interval of t2
		for i in range(len(idx)):
			if pm[i]==0:
				pvinterp=np.concatenate([pvinterp,[pv2[idx[i]]]])
				#pvinterp_at=np.concatenate([pvinterp_at,[pv_at[idx[i]]]])
			elif pm[i]<0:
				pvinterp=np.concatenate([pvinterp,[pv2[idx[i]-1]+(pv2[idx[i]]-pv2[idx[i]-1])*(t[i]-t2[idx[i]-1])/(t2[idx[i]]-t2[idx[i]-1])]])
				#pvinterp_at=np.concatenate([pvinterp_at,[pv_at[idx[i]-1]+(pv_at[idx[i]]-pv_at[idx[i]-1])*(t[i]-t2[idx[i]-1])/(t2[idx[i]]-t2[idx[i]-1])]])
			elif pm[i]>0:
				pvinterp=np.concatenate([pvinterp,[pv2[idx[i]]+(pv2[idx[i]+1]-pv2[idx[i]])*(t[i]-t2[idx[i]])/(t2[idx[i]+1]-t2[idx[i]])]])
				#pvinterp_at=np.concatenate([pvinterp_at,[pv_at[idx[i]]+(pv_at[idx[i]+1]-pv_at[idx[i]])*(t[i]-t2[idx[i]])/(t2[idx[i]+1]-t2[idx[i]])]])
		globalfit=np.concatenate([globalfit,pvinterp])
	return globalfit

def ccrwrs_sat_conv_lin(t,c,a0,bsat,tau1,tau2,t0,sigma,rangefactor=.5):
	"""
	t0=-0
	bsat=1
	tau1=120
	tau2=10700
	sigma=15
	t=np.linspace(-1000,8000,46200)
	idt0=(np.abs(t - t0)).argmin()
	print(t[idt0])
	print(idt0)
	print(len(t[idt0:]))
	powerseries= np.logspace(-2,2, num=11,base=np.e)
	fig, ax = plt.subplots()
	colors = [plt.cm.jet(i) for i in np.linspace(0, 1,len(powerseries))]
	ax.set_prop_cycle('color', colors)
	for i in powerseries:
		label0=str(i)[0:4]
		b =ps.ccrwrs_sat_conv_interp(t,i,bsat,tau1,tau2,t0,sigma,rangefactor=0.30557)
		plt.plot(t,b,label=label0)
	plt.rcParams.update({'font.size': 10})  # increase the font size
	plt.xlabel("time Delay (ps")
	plt.ylabel("Population of 'B'")
	plt.legend(loc='best')
	#plt.xlim(-202,-198)
	#plt.xticks([10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000]) 
	plt.show()
	"""
	if t[-1]<t[0]:
		print("XXXXX ERRROR XXXXX time array must be increasing for convolve to work")
	#buffer the time domain to be twice as big so we don't get edge effects.
	idt0=(np.abs(t - t0)).argmin()
	gauss=np.exp(-((t)/sigma)**2/2)/sigma
	s =np.insert(odeint(isat, [a0,0,bsat,tau1,tau2], t[idt0:])[:,1],0,np.zeros(idt0))
	ssgauss=c*np.convolve(s, gauss, mode="full")[int(idt0):int(len(s)+idt0)]
	return ssgauss

def ccrwrs_sat_conv_linb(t,c,a0,bsat,tau1,tau2,t0,sigma):
	"""
	t0=-0
	bsat=1
	tau1=120
	tau2=10700
	sigma=15
	t=np.linspace(-1000,8000,46200)
	idt0=(np.abs(t - t0)).argmin()
	print(t[idt0])
	print(idt0)
	print(len(t[idt0:]))
	powerseries= np.logspace(-2,2, num=11,base=np.e)
	fig, ax = plt.subplots()
	colors = [plt.cm.jet(i) for i in np.linspace(0, 1,len(powerseries))]
	ax.set_prop_cycle('color', colors)
	for i in powerseries:
		label0=str(i)[0:4]
		b =ps.ccrwrs_sat_conv_interp(t,i,bsat,tau1,tau2,t0,sigma,rangefactor=0.30557)
		plt.plot(t,b,label=label0)
	plt.rcParams.update({'font.size': 10})  # increase the font size
	plt.xlabel("time Delay (ps")
	plt.ylabel("Population of 'B'")
	plt.legend(loc='best')
	#plt.xlim(-202,-198)
	#plt.xticks([10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000]) 
	plt.show()
	"""
	if t[-1]<t[0]:
		print("XXXXX ERRROR XXXXX time array must be increasing for convolve to work")
	#buffer the time domain to be twice as big so we don't get edge effects.
	mid=(t[-1]+t[0])/2
	r=t[-1]-t[0]
	t2=np.linspace(mid-r/2,mid+r/2,2*len(t))
	idt0=(np.abs(t2 - t0)).argmin()
	gauss=np.exp(-((t2)/sigma)**2/2)/sigma
	s =np.insert(odeint(isat, [a0,0,bsat,tau1,tau2], t2[idt0:])[:,1],0,np.zeros(idt0))
	ssgauss=c*np.convolve(s, gauss, mode="full")[int(idt0):int(len(s)+idt0):2]
	return ssgauss

def ccrwrs_sat_conv_interp(t,c,a0,bsat,tau1,tau2,t0,sigma):
	"""
t0=-0
bsat=1
tau1=120
tau2=10700
sigma=15
t=np.linspace(-1000,8000,46200)
idt0=(np.abs(t - t0)).argmin()
print(t[idt0])
print(idt0)
print(len(t[idt0:]))
powerseries= np.logspace(-2,2, num=11,base=np.e)
fig, ax = plt.subplots()
colors = [plt.cm.jet(i) for i in np.linspace(0, 1,len(powerseries))]
ax.set_prop_cycle('color', colors)
for i in powerseries:
	label0=str(i)[0:4]
	b =ps.ccrwrs_sat_conv_interp(t,i,bsat,tau1,tau2,t0,sigma,=0.30557rangefactor)
	plt.plot(t,b,label=label0)
plt.rcParams.update({'font.size': 10})  # increase the font size
plt.xlabel("time Delay (ps")
plt.ylabel("Population of 'B'")
plt.legend(loc='best')
#plt.xlim(-202,-198)
#plt.xticks([10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000]) 
plt.show()
	"""
	if t[-1]<t[0]:
		print("XXXXX ERRROR XXXXX time array must be increasing for convolve to work")
	numpoints= 92400
	#buffer the time domain to be twice as big so we don't get edge effects.
	mid=(t[-1]+t[0])/2
	r=t[-1]-t[0]
	t2=np.linspace(mid-r/2,mid+r/2,2*numpoints)
	idt0=(np.abs(t2 - t0)).argmin()
	gauss=np.exp(-((t2)/sigma)**2/2)/sigma
	s =np.insert(odeint(isat, [a0,0,bsat,tau1,tau2], t2[idt0:])[:,1],0,np.zeros(idt0))
	ssgauss=c*np.convolve(s, gauss, mode="full")[int(idt0):int(len(s)+idt0):2]
	idx=[(np.abs(t2 - tx)).argmin() for tx in t]
	pm=[np.sign(t2[idx[j]]-t[j]) for j in range(len(idx))]
	s3=[]
	#linearly interpolate ssgauss between interval of t2
	for i in range(len(idx)):
		if pm[i]==0:
			s3.append(ssgauss[idx[i]])
		elif pm[i]<0:
			s3.append(ssgauss[idx[i]-1]+(ssgauss[idx[i]]-ssgauss[idx[i]-1])*(t[i]-t2[idx[i]-1])/(t2[idx[i]]-t2[idx[i]-1]))
		elif pm[i]>0:
			s3.append(ssgauss[idx[i]]+(ssgauss[idx[i]+1]-ssgauss[idx[i]])*(t[i]-t2[idx[i]])/(t2[idx[i]+1]-t2[idx[i]]))
	return s3

def peep0(t,p0):
	"""
	photoelectron electronphoton single exponentials
	"""
	t0=p0[0]
	a1=p0[1]
	tau1=p0[2]
	a2=p0[3]
	tau2=p0[4]
	sigma= p0[5]
	s=(a1*np.heaviside(t0-t, 0)*np.exp((t-t0)/tau1)+a2*np.heaviside(t-t0, 0)*np.exp(-(t-t0)/tau2))
	return s

def peep_conv_interp(t,p0):
	"""
	photoelectron electronphoton single exponentials
	BEST CONVOLUTION FUNCTION!
	Example:
	timepoints=np.linspace(-5000,5000,1000)
	sigmas=np.array([5,10,50,100,500])
	for s in sigmas:
		p0=np.array([1000,-1.,150,1,4000.,s])
		f=ps.peep_conv_interp(timepoints,p0)
		f2=ps.peep_conv_interp(tp0,p0)
		plt.plot(timepoints, f,tp0, f2, marker='o',)
	"""
	t0=p0[0]
	a1=p0[1]
	tau1=p0[2]
	a2=p0[3]
	tau2=p0[4]
	sigma=p0[5]
	if t[-1]<t[0]:
		print("XXXXX ERRROR XXXXX time array must be increasing odeint to work.")
	numpoints= 9240*5
	#buffer the time domain to be twice as big so we don't get edge effects.
	t2=np.linspace(-18000,18000,numpoints)
	t3=t2+t0
	gauss=np.exp(-((t2)/sigma)**2/2)/sigma/2/np.pi
	s=(a1*np.heaviside(-t2, 0)*np.exp((t2)/tau1)+a2*np.heaviside(t2, 0)*np.exp(-(t2)/tau2))
	ssgauss=np.convolve(s, gauss, mode="full")[::2]
	idx=[(np.abs(t3 - tx)).argmin() for tx in t]
	pm=[np.sign(t3[idx[j]]-t[j]) for j in range(len(idx))]
	s3=[]
	#linearly interpolate ssgauss between interval of t2
	for i in range(len(idx)):
		if pm[i]==0:
			s3.append(ssgauss[idx[i]])
		elif pm[i]<0:
			s3.append(ssgauss[idx[i]-1]+(ssgauss[idx[i]]-ssgauss[idx[i]-1])*(t[i]-t3[idx[i]-1])/(t3[idx[i]]-t3[idx[i]-1]))
		elif pm[i]>0:
			s3.append(ssgauss[idx[i]]+(ssgauss[idx[i]+1]-ssgauss[idx[i]])*(t[i]-t3[idx[i]])/(t3[idx[i]+1]-t3[idx[i]]))
	return s3

def negtimeconv2(t,hnu0,scalar,t0,sigma0,tau1,tau2,tau3,esat,gamma,alpha):
	if t[-1]<t[0]:
		print("XXXXX ERRROR XXXXX time array must be increasing odeint to work.")
	print("apple")
	numpoints= 90000
	#buffer the time domain to be twice as big so we don't get edge effects.
	t2=np.linspace(-10000,9999,20000)
	t3=t2-t2[0]
	#idt0=(np.abs(t2 - t0)).argmin()
	s =odeint(e_gen_sat3, [0,0,0,0], t3,args=(hnu0,t0+t3[0]-t2[0],sigma0,tau1,tau2,esat,gamma,alpha),hmax=1.0)
	exp=np.exp(((t2)/tau3))*np.heaviside((-t2+t0), 0)
	#gauss=np.exp(-((t2)/sigma)**2/2)/sigma/2/np.pi
	#ps.plotkinetictrace(exp,t2,save=1,destination="testhevix.txt",logx=0)
	#ps.plotkinetictrace(s[:,3],t2,save=1,destination="testhevix.txt",logx=0)
	#print("xxxxxxx")
	#print(len(exp))
	#print(type(exp))
	#print(np.shape(s[:,3]))
	#print(type(s))
	ssexp=np.convolve(s[:,3], exp, mode="full")
	s2=ssexp[9999:29999]
	#print(np.shape(ssexp[9999:29999]))
	#ps.plotkinetictrace(ssexp[9999:29999],t2,save=1,destination="testhevix.txt",logx=0)
	#print("xxxxxxx")
	idx=[(np.abs(t2 - tx)).argmin() for tx in t]
	pm=[np.sign(t2[idx[j]]) for j in range(len(idx))]
	#print("idx:",idx)
	#print("t",t)
	#print("t2[idx]",t2[idx])
	#print("pm:",pm)
	#s3=[]
	s3=np.array([s2[idx[i]] for i in range(len(idx))])
	#linearly interpolate ssexp between interval of t2
	#for i in range(len(idx)):
	#	print(s2[idx[i]])
	#	if pm[i]==0:
	#		s3.append(s2[idx[i]])
	#	elif pm[i]<0:
	#		s3.append(s2[idx[i]-1])
	#	elif pm[i]>0:
	#		s3.append(s2[idx[i]])
	# for i in range(len(idx)):
	# 	if pm[i]==0:
	# 		np.concatenate([s3,[]])
	# 	elif pm[i]<0:
	# 		np.concatenate([s3,[s2[idx[i]-1]]])
	# 	elif pm[i]>0:
	# 		np.concatenate([s3,[s2[idx[i]]]])
	return s3

def negtimeconv2step(t,hnu0,scalar,t0,sigma0,tau1,tau2,tau3,esat,gamma,alpha,xi):
	if t[-1]<t[0]:
		print("XXXXX ERRROR XXXXX time array must be increasing odeint to work.")
	numpoints= 90000
	#buffer the time domain to be twice as big so we don't get edge effects.
	t2=np.linspace(-10000,9999,20000)
	t3=t2-t2[0]
	#idt0=(np.abs(t2 - t0)).argmin()
	s =odeint(e_gen_sat4, [0,0,0,0], t3,args=(hnu0,t0+t3[0]-t2[0],sigma0,tau1,tau2,esat,gamma,alpha,xi),hmax=1.0)
	exp=np.exp(((t2)/tau3))*np.heaviside((-t2+t0), 0)
	#ps.plotkinetictrace(exp,t2,save=1,destination="testhevix.txt",logx=0)
	#ps.plotkinetictrace(s[:,3],t2,save=1,destination="testhevix.txt",logx=0)
	#print("xxxxxxx")
	#print(len(exp))
	#print(type(exp))
	#print(np.shape(s[:,3]))
	#print(type(s))
	ssexp=np.convolve(s[:,3], exp, mode="full")
	s2=ssexp[9999:29999]
	#print(np.shape(ssexp[9999:29999]))
	#ps.plotkinetictrace(ssexp[9999:29999],t2,save=1,destination="testhevix.txt",logx=0)
	#print("xxxxxxx")
	idx=[(np.abs(t2 - tx)).argmin() for tx in t]
	pm=[np.sign(t2[idx[j]]) for j in range(len(idx))]
	#print("idx:",idx)
	#print("t",t)
	#print("t2[idx]",t2[idx])
	#print("pm:",pm)
	#s3=[]
	s3=np.array([s2[idx[i]] for i in range(len(idx))])
	#linearly interpolate ssexp between interval of t2
	#for i in range(len(idx)):
	#	print(s2[idx[i]])
	#	if pm[i]==0:
	#		s3.append(s2[idx[i]])
	#	elif pm[i]<0:
	#		s3.append(s2[idx[i]-1])
	#	elif pm[i]>0:
	#		s3.append(s2[idx[i]])
	# for i in range(len(idx)):
	# 	if pm[i]==0:
	# 		np.concatenate([s3,[]])
	# 	elif pm[i]<0:
	# 		np.concatenate([s3,[s2[idx[i]-1]]])
	# 	elif pm[i]>0:
	# 		np.concatenate([s3,[s2[idx[i]]]])
	return s3



def peep_conv_interp_b(t,t0,a1,tau1,a2,tau2,sigma):
	"""
	photoelectron electronphoton single exponentials
	BEST CONVOLUTION FUNCTION!
	Example:
	timepoints=np.linspace(-5000,5000,1000)
	sigmas=np.array([5,10,50,100,500])
	for s in sigmas:
		p0=np.array([1000,-1.,150,1,4000.,s])
		f=ps.peep_conv_interp(timepoints,p0)
		f2=ps.peep_conv_interp(tp0,p0)
		plt.plot(timepoints, f,tp0, f2, marker='o',)
	"""
	if t[-1]<t[0]:
		print("XXXXX ERRROR XXXXX time array must be increasing odeint to work.")
	numpoints= 9240*5
	#buffer the time domain to be twice as big so we don't get edge effects.
	t2=np.linspace(-18000,18000,numpoints)
	t3=t2+t0
	gauss=np.exp(-((t2)/sigma)**2/2)/sigma/2/np.pi
	s=np.array((a1*np.heaviside(-t2, 0)*np.exp(np.minimum(t2,0)/tau1)+a2*np.heaviside(t2, 0)*np.exp(-(np.maximum(t2,0))/tau2)))
	ssgauss=np.convolve(s, gauss, mode="full")[::2]
	idx=[(np.abs(t3 - tx)).argmin() for tx in t]
	pm=[np.sign(t3[idx[j]]-t[j]) for j in range(len(idx))]
	s3=[]
	#linearly interpolate ssgauss between interval of t2
	for i in range(len(idx)):
		if pm[i]==0:
			s3.append(ssgauss[idx[i]])
		elif pm[i]<0:
			s3.append(ssgauss[idx[i]-1]+(ssgauss[idx[i]]-ssgauss[idx[i]-1])*(t[i]-t3[idx[i]-1])/(t3[idx[i]]-t3[idx[i]-1]))
		elif pm[i]>0:
			s3.append(ssgauss[idx[i]]+(ssgauss[idx[i]+1]-ssgauss[idx[i]])*(t[i]-t3[idx[i]])/(t3[idx[i]+1]-t3[idx[i]]))
	return s3

def peep_dblexp_conv_interp(t,p0):
	"""
	photoelectron electronphoton single exponentials
	BEST CONVOLUTION FUNCTION!
	Example:
	timepoints=np.linspace(-5000,5000,1000)
	sigmas=np.array([5,10,50,100,500])
	for s in sigmas:
		p0=np.array([1000,-1.,150,1,4000.,s])
		f=ps.peep_conv_interp(timepoints,p0)
		f2=ps.peep_conv_interp(tp0,p0)
		plt.plot(timepoints, f,tp0, f2, marker='o',)
	"""
	t0=p0[0]
	a1=p0[1]
	tau1=p0[2]
	a2=p0[3]
	tau2=p0[4]
	a3=p0[4]
	tau3=p0[5]
	sigma=p0[6]
	if t[-1]<t[0]:
		print("XXXXX ERRROR XXXXX time array must be increasing odeint to work.")
	numpoints= 9240*5
	#buffer the time domain to be twice as big so we don't get edge effects.
	t2=np.linspace(-18000,18000,numpoints)
	t3=t2+t0
	gauss=np.exp(-((t2)/sigma)**2/2)/sigma/2/np.pi
	s=(a1*np.heaviside(-t2, 0)*np.exp((t2)/tau1)+a2*np.heaviside(t2, 0)*np.exp(-(t2)/tau2)+a3*np.heaviside(t2, 0)*np.exp(-(t2)/tau3))
	ssgauss=np.convolve(s, gauss, mode="full")[::2]
	idx=[(np.abs(t3 - tx)).argmin() for tx in t]
	pm=[np.sign(t3[idx[j]]-t[j]) for j in range(len(idx))]
	s3=[]
	#linearly interpolate ssgauss between interval of t2
	for i in range(len(idx)):
		if pm[i]==0:
			s3.append(ssgauss[idx[i]])
		elif pm[i]<0:
			s3.append(ssgauss[idx[i]-1]+(ssgauss[idx[i]]-ssgauss[idx[i]-1])*(t[i]-t3[idx[i]-1])/(t3[idx[i]]-t3[idx[i]-1]))
		elif pm[i]>0:
			s3.append(ssgauss[idx[i]]+(ssgauss[idx[i]+1]-ssgauss[idx[i]])*(t[i]-t3[idx[i]])/(t3[idx[i]+1]-t3[idx[i]]))
	return s3

def peep_dblexp_conv_interp_b(t,t0,a1,tau1,a2,tau2,a3,tau3,sigma):
	"""
	photoelectron electronphoton single exponentials
	BEST CONVOLUTION FUNCTION!
	Example:
	timepoints=np.linspace(-5000,5000,1000)
	sigmas=np.array([5,10,50,100,500])
	for s in sigmas:
		p0=np.array([1000,-1.,150,1,4000.,s])
		f=ps.peep_conv_interp(timepoints,p0)
		f2=ps.peep_conv_interp(tp0,p0)
		plt.plot(timepoints, f,tp0, f2, marker='o',)
	"""
	if t[-1]<t[0]:
		print("XXXXX ERRROR XXXXX time array must be increasing odeint to work.")
	numpoints= 9240*5
	#buffer the time domain to be twice as big so we don't get edge effects.
	t2=np.linspace(-18000,18000,numpoints)
	t3=t2+t0
	gauss=np.exp(-((t2)/sigma)**2/2)/sigma/2/np.pi
	s=(a1*np.heaviside(-t2, 0)*np.exp((t2)/tau1)+a2*np.heaviside(t2, 0)*np.exp(-(t2)/tau2)+a3*np.heaviside(t2, 0)*np.exp(-(t2)/tau3))
	ssgauss=np.convolve(s, gauss, mode="full")[::2]
	idx=[(np.abs(t3 - tx)).argmin() for tx in t]
	pm=[np.sign(t3[idx[j]]-t[j]) for j in range(len(idx))]
	s3=[]
	#linearly interpolate ssgauss between interval of t2
	for i in range(len(idx)):
		if pm[i]==0:
			s3.append(ssgauss[idx[i]])
		elif pm[i]<0:
			s3.append(ssgauss[idx[i]-1]+(ssgauss[idx[i]]-ssgauss[idx[i]-1])*(t[i]-t3[idx[i]-1])/(t3[idx[i]]-t3[idx[i]-1]))
		elif pm[i]>0:
			s3.append(ssgauss[idx[i]]+(ssgauss[idx[i]+1]-ssgauss[idx[i]])*(t[i]-t3[idx[i]])/(t3[idx[i]+1]-t3[idx[i]]))
	return s3


def interfacialsaturation(y,t):
	"""
	Define a function which calculates the derivative
	Example Code
	t=np.linspace(0,8000,20000)
	powerseries= np.logspace(-2,2, num=11,base=np.e)
	fig, ax = plt.subplots()
	colors = [plt.cm.jet(i) for i in np.linspace(0, 1,len(powerseries))]
	ax.set_prop_cycle('color', colors)
	for i in powerseries:
		label0=str(i)[0:4]
		ys = odeint(interfacialsaturation, [i,0], t)
		b=ys[:,1]
		plt.semilogx(t,b,label=label0)
	plt.rcParams.update({'font.size': 10})  # increase the font size
	plt.xlabel("time Delay (ps")
	plt.ylabel("Population of 'B'")
	plt.xlim(8,8000)
	plt.legend(loc='best')
	plt.xticks([10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000]) 
	plt.show()
	"""
	k1=1/1000
	k2=1/2000
	k3=1/2000
	bsat=1
	a=y[0]
	b=y[1]
	da=-k1*a+k1*a*b/bsat-k2*a
	db=k1*a-k1*a*b/bsat-k3*b
	#da=-k1*a-k2*a
	#db=k1*a-k3*b
	return [da,db]


def getftmat(timeaxis,fluaxis,c,bsat,tau1,tau2,t0,sigma,save=1,destination="ftmat.txt"):
	flumat=np.array([], dtype=np.int64).reshape(0,len(timeaxis))
	counter =0
	for i in fluaxis:
		#print(counter)
		flumat=np.vstack((flumat,ccrwrs_sat_conv_linb(timeaxis,c,i,bsat,tau1,tau2,t0,sigma)))
		counter+=1
	np.savetxt(destination,np.vstack((timeaxis,flumat)),delimiter='\t', newline='\n', fmt='%1.18f')
	return pd.DataFrame(data=flumat)

def getftmat2(timeaxis,fluaxis,p0,save=1,destination="ftmat.txt"):
	#Note we need to drop the first element of the fit function so that we can get the values from the fluaxis
	p1=np.delete(p0,0)
	flumat=np.array([], dtype=np.int64).reshape(0,len(timeaxis))
	for i in fluaxis:
		#print(counter)
		flumat=np.vstack((flumat,eh_gen_sat_interp_attenuation(timeaxis,i,*p1)))
	np.savetxt(destination,np.vstack((timeaxis,flumat)),delimiter='\t', newline='\n', fmt='%1.18f')
	return pd.DataFrame(data=flumat)

def getftmat3(timeaxis,fluaxis,p0,save=1,destination="ftmat.txt"):
	#Note we need to drop the first element of the fit function so that we can get the values from the fluaxis
	p1=np.delete(p0,0)
	flumat=np.array([], dtype=np.int64).reshape(0,len(timeaxis))
	for i in fluaxis:
		#print(counter)
		flumat=np.vstack((flumat,e_gen_sat_interp(timeaxis,i,*p1)))
	np.savetxt(destination,np.vstack((timeaxis,flumat)),delimiter='\t', newline='\n', fmt='%1.18f')
	return pd.DataFrame(data=flumat)

def getftmat4(timeaxis,fluaxis,p0,save=1,destination="ftmat.txt"):
	#Note we need to drop the first element of the fit function so that we can get the values from the fluaxis
	p1=np.delete(p0,0)
	flumat=np.array([], dtype=np.int64).reshape(0,len(timeaxis))
	for i in fluaxis:
		#print(counter)
		flumat=np.vstack((flumat,e_gen_sat_interp3(timeaxis,i,*p1)))
	np.savetxt(destination,np.vstack((timeaxis,flumat)),delimiter='\t', newline='\n', fmt='%1.18f')
	return pd.DataFrame(data=flumat)

def getftmat5(timeaxis,fluaxis,p0,save=1,destination="ftmat.txt"):
	#Note we need to drop the first element of the fit function so that we can get the values from the fluaxis
	p1=np.delete(p0,0)
	flumat=np.array([], dtype=np.int64).reshape(0,len(timeaxis))
	for i in fluaxis:
		#print(counter)
		flumat=np.vstack((flumat,negtimeconv(timeaxis,i,*p1)))
	np.savetxt(destination,np.vstack((timeaxis,flumat)),delimiter='\t', newline='\n', fmt='%1.18f')
	return pd.DataFrame(data=flumat)

def getfluindexmat(profilemat,fluaxis):
	idf=np.reshape(np.array([(np.abs(fluaxis - q)).argmin() for q in np.ravel(profilemat)]),np.shape(profilemat))
	return idf

def gettimeindex(timepoints,timeaxis):
	idt=[(np.abs(timeaxis - tx)).argmin() for tx in timepoints]
	return idt

def suemsim(ftmat,profilemat,timepoints,fluaxis,timeaxis,destinationbase="suemsim"):
	idf=getfluindexmat(profilemat,fluaxis)
	idt=gettimeindex(timepoints,timeaxis)
	counter=0
	for i in idt:
		tempsuem= np.flip(np.reshape(np.array([ftmat.iloc[p,i] for p in np.ravel(idf)]),np.shape(idf)),0)
		np.savetxt(destinationbase+"_t"+str(timepoints[counter])+".txt",tempsuem)
		counter+=1
	return

def ccrwrs_conv_interp_global(t,A,tau1,tau2):
	"""
	import matplotlib.pyplot as plt
	tp0=ps.loadtimepoints("timepoints20200131181703.txt")
	timepoints=np.linspace(tp0[0],tp0[-1],5000)
	p0=[-.08,10000000,150,700,10]
	f=ps.ccrwrs_conv_lin(timepoints,-.08,10000000,150,7000,1000)
	f2=ps.ccrwrs_conva(tp0,-.08,10000000,150,7000,1000)
	f3=ps.ccrwrs_conv_interp(tp0,-.08,10000000,150,7000,1000)
	if 1:
		plt.plot(timepoints, f,tp0, f2,tp0, f3, marker='o',)
		plt.show()
	"""
	fluence=np.array([4.977398588281487719e+03,4.404647532228844284e+03,3.448158418366845581e+03,2.388590356149985382e+03,1.464142202073505359e+03,7.944616977665792774e+02,3.816054818303753677e+02,1.620247113588931427e+02,6.087466625122205954e+01,2.025216173371581263e+01,5.954448440814415378e+00,1.547548705936377944e+00,3.562716592149722805e-01,7.260799262078454497e-02,1.327004221807980867e-02,2.161936070359874763e-03,3.033134915878706039e-04,3.791413449104685611e-05,4.298452218174583343e-06,4.250790131055891801e-07,3.683883678578665888e-08,2.834347159838135233e-09,2.039871627004095886e-10,1.234124411359116268e-11,6.697065373958510758e-13])
	t0=-660.
	sigma=9.2*2**.5
	numpoints=5000
	lambdax=100000
	t1=t[0:int(len(t)/len(fluence2))]
	#print(t[len(t)/len(fluence2)])
	t2=np.linspace(t1[0],t1[-1],5000)
	gauss=np.exp(-((t2-t0)/sigma)**2/2)/sigma
	s2=[]
	for flu in fluence2:
		s =A*(1-np.exp(-flu/rho2))*(np.exp(-(t2-t0)/tau1)-np.exp(-(t2-t0)/tau2))*np.heaviside((t2-t0), 0)
		ssgauss=np.convolve(s, gauss, mode="full")[0:len(t2)]
		idx=[(np.abs(t2 - tx)).argmin() for tx in t1]
		pm=[np.sign(t2[idx[j]]-t1[j]) for j in range(len(idx))]
		#linearly interpolate ssgauss between interval of t2
		for i in range(len(idx)):
			if pm[i]==0:
				s2.append(ssgauss[idx[i]])
			elif pm[i]<0:
				s2.append(ssgauss[idx[i]-1]+(ssgauss[idx[i]]-ssgauss[idx[i]-1])*(t1[i]-t2[idx[i]-1])/(t2[idx[i]]-t2[idx[i]-1]))
			elif pm[i]>0:
				s2.append(ssgauss[idx[i]]+(ssgauss[idx[i]+1]-ssgauss[idx[i]])*(t1[i]-t2[idx[i]])/(t2[idx[i]+1]-t2[idx[i]]))
	return s2

def fluenceresponse0(A,rho,f):
	return A*(1-np.exp(-f/rho))

def beamprofilefluence(power,radiusx=0,radiusy=0.,wx=84./2.,wy=49./2,mux=0,muy=0,l=0.515,depth=10**4,reprate=1e6,materialreflectivity=0):
	"""
	module for calculating the beam profile on the sample in units of 10**15 photons/cm2 if is not specified or 10**15 photons/cm**3 or petaphotons/cm3 if depth is specified.
	all inputs are in units of microns and milliwats.
	:param float: power: unit mW power before the chamber
	:param float: radiusx: unit um position in beam x offset from center position
	:param float: radiusy: unit um position in beam y offset from center position
	:param float: wx: unit um 1/e2 radius of beam in x
	:param float: wy: unit um 1/e2 radius of beam in y
	:param float: reprate: unit hertz rep rate of pump. Note could be 2e6
	:param float: mux: unit um center position in beam x
	:param float: muy: unit um center position in beam y
	:param float: l: unit um wavelength of light used
	:param float: depth: unit um attenuation depth of light in the sample
	Example:
	returns the fluence in units of joules /um^2 if l=0 or photons /um if l is not 0. It takes power in units of mW
	"""
	photons=energyinchamber(power,materialreflectivity=materialreflectivity,reprate=reprate)/energyofphoton(l)
	return 2/3.1415259*photons/10**15/((depth*10**-4)*(wx*10**-4)*(wy*10**-4))*np.exp(-2*(((((radiusx-mux)*10**-4)**2)/((wx*10**-4)**2))+(((radiusy-muy)*10**-4)**2)/((wy*10**-4)**2)))

def beamprofilefluence_rot(power,radiusx=0,radiusy=0.,wx=84./2.,wy=49./2,mux=0,muy=0,l=0.515,depth=10**4,theta=0,reprate=1e6,radians=0,materialreflectivity=0):
	"""
	module for calculating the beam profile on the sample in units of 10**15 photons/cm2 if is not specified or 10**15 photons/cm**3 or petaphotons/cm3 if depth is specified.
	all inputs are in units of microns and milliwats.
	:param float: power: unit mW power before the chamber
	:param float: radiusx: unit um position in beam x offset from center position
	:param float: radiusy: unit um position in beam y offset from center position
	:param float: wx: unit um 1/e2 radius of beam in x
	:param float: wy: unit um 1/e2 radius of beam in y
	:param float: reprate: unit hertz rep rate of pump. Note could be 2e6
	:param float: mux: unit um center position in beam x
	:param float: muy: unit um center position in beam y
	:param float: l: unit um wavelength of light used
	:param float: depth: unit um attenuation depth of light in the sample
	Example:
	returns the fluence in units of joules /um^2 if l=0 or photons /um if l is not 0. It takes power in units of mW
	"""
	if(radians==0):
		theta=theta*3.1415259/180
	photons=energyinchamber(power,materialreflectivity=materialreflectivity,reprate=reprate)/energyofphoton(l)
	return 2/3.1415259*photons/10**15/((depth*10**-4)*(wx*10**-4)*(wy*10**-4))*np.exp(-2*(   ((( (np.cos(theta)*((radiusx-mux)*10**-4)-(np.sin(theta)*((radiusy-muy)*10**-4)))**2)  /   ((wx*10**-4)**2))+  ((np.sin(theta)*((radiusx-mux)*10**-4)+(np.cos(theta)*((radiusy-muy)*10**-4)))**2)/((wy*10**-4)**2))))

def beamprofileexposure_rot(power,radiusx=0,radiusy=0.,wx=84./2.,wy=49./2,mux=0,muy=0,theta=0,reprate=1e6,radians=0,materialreflectivity=0):
	"""
	module for calculating the beam profile on the sample in units of mJ/cm.
	all inputs are in units of microns and milliwats.
	:param float: power: unit mW power before the chamber
	:param float: radiusx: unit um position in beam x offset from center position
	:param float: radiusy: unit um position in beam y offset from center position
	:param float: wx: unit um 1/e2 radius of beam in x
	:param float: wy: unit um 1/e2 radius of beam in y
	:param float: reprate: unit hertz rep rate of pump. Note could be 2e6
	:param float: mux: unit um center position in beam x
	:param float: muy: unit um center position in beam y
	:param float: l: unit um wavelength of light used
	:param float: depth: unit um attenuation depth of light in the sample
	Example:
	returns the fluence in units of joules /um^2 if l=0 or photons /um if l is not 0. It takes power in units of mW
	"""
	if(radians==0):
		theta=theta*3.1415259/180
	E=energyinchamber(power,materialreflectivity=materialreflectivity,reprate=reprate)
	return 1000*2/3.1415259*E/((wx*10**-4)*(wy*10**-4))*np.exp(-2*(   ((( (np.cos(theta)*((radiusx-mux)*10**-4)-(np.sin(theta)*((radiusy-muy)*10**-4)))**2)  /   ((wx*10**-4)**2))+  ((np.sin(theta)*((radiusx-mux)*10**-4)+(np.cos(theta)*((radiusy-muy)*10**-4)))**2)/((wy*10**-4)**2))))

def beamprofileexposure_rot_hd(power,radiusx=0,radiusy=0.,wx=38.33,wy=22.964,fitp1=np.array([-3.56,0.01949,103.5,-28.814,.161676,21.14,-.212,.97575,38.33]),mux=0,muy=0,theta=0,reprate=1e6,radians=0,materialreflectivity=0):
	"""
	module for calculating the beam profile on the sample in units of mJ/cm.
	all inputs are in units of microns and milliwats.
	:param float: power: unit mW power before the chamber
	:param float: radiusx: unit um position in beam x offset from center position
	:param float: radiusy: unit um position in beam y offset from center position
	:param float: wx: unit um 1/e2 radius of beam in x
	:param float: wy: unit um 1/e2 radius of beam in y
	:param float: reprate: unit hertz rep rate of pump. Note could be 2e6
	:param float: mux: unit um center position in beam x
	:param float: muy: unit um center position in beam y
	:param float: l: unit um wavelength of light used
	:param float: depth: unit um attenuation depth of light in the sample
	Example:
	returns the fluence in units of joules /um^2 if l=0 or photons /um if l is not 0. It takes power in units of mW
	"""
	print(fitp1)
	if(radians==0):
		theta=theta*3.1415259/180
	E=energyinchamber(power,materialreflectivity=materialreflectivity,reprate=reprate)
	peakflu=1000*2/3.1415259*E/((wx*10**-4)*(wy*10**-4))
	lnx1=-2*( (np.cos(theta)*((radiusx-fitp1[0]-mux)*10**-4)-(np.sin(theta)*((radiusy-fitp1[0]-muy)*10**-4)))**2)  /   ((fitp1[2]*10**-4)**2) 
	lnx2=-2*( (np.cos(theta)*((radiusx-fitp1[3]-mux)*10**-4)-(np.sin(theta)*((radiusy-fitp1[3]-muy)*10**-4)))**2)  /   ((fitp1[5]*10**-4)**2) 
	lnx3=-2*( (np.cos(theta)*((radiusx-fitp1[6]-mux)*10**-4)-(np.sin(theta)*((radiusy-fitp1[6]-muy)*10**-4)))**2)  /   ((fitp1[8]*10**-4)**2) 
	lny=-2*( ( (np.sin(theta)*((radiusx-mux)*10**-4)+(np.cos(theta)*((radiusy-muy)*10**-4)))**2)  /   ((wy*10**-4)**2) )
	return peakflu*(fitp1[1]*np.exp(lnx1+lny)+fitp1[4]*np.exp(lnx2+lny)+fitp1[7]*np.exp(lnx3+lny))


def getfluaxis(mag,power,mux=0,muy=0,materialreflectivity=0.0,reprate=1E6,wx=42,wy=24.5):
	shiftx,shifty=getpositionaxis(mag, xpixel=738,ypixel=485,origintopleft=0)
	peakflu=fluenceinchamber(power,materialreflectivity=materialreflectivity,reprate=reprate,wx=wx,wy=wy)
	flux=peakflu*np.exp(-2*(shiftx-mux)**2/wx**2)
	fluy=peakflu*np.exp(-2*(shifty-muy)**2/wy**2)
	return flux,fluy

def getfluaxistriplegauss(mag,power,mux=0,muy=0,materialreflectivity=0.0,reprate=1E6,fitp1=np.array([-3.56,0.01949,103.5,-28.814,.161676,21.14,-.212,.97575,38.33]),wx=38.33,wy=22.964):
	shiftx,shifty=getpositionaxis(mag, xpixel=738,ypixel=485,origintopleft=0)
	peakflu=fluenceinchamber(power,materialreflectivity=materialreflectivity,reprate=reprate,wx=wx,wy=wy)
	print("amps:", fitp1[1],fitp1[4],fitp1[7])
	print("1/e2:", fitp1[2],fitp1[5],fitp1[8])
	print("centers:", fitp1[0],fitp1[3],fitp1[6])
	flux=peakflu*(fitp1[1]*np.exp(-2*(shiftx-fitp1[0]-mux)**2/fitp1[2]**2)+fitp1[4]*np.exp(-2*(shiftx-fitp1[3]-mux)**2/fitp1[5]**2)+fitp1[7]*np.exp(-2*(shiftx-fitp1[6]-mux)**2/fitp1[8]**2))
	fluy=peakflu*np.exp(-2*(shifty-muy)**2/wy**2)
	return flux,fluy

def getexposuremat_hd(power,mag,wx,wy,materialreflectivity=0.00,mux=0,muy=0,reprate=1e6,xpixel=738,ypixel=485,theta=0,radians=0, save=0,destination="beamprofile.txt",fitp1=np.array([-3.56,0.01949,103.5,-28.814,.161676,21.14,-.212,.97575,38.33])):
	print(theta)
	shiftx,shifty=getpositionaxis(mag, xpixel=xpixel,ypixel=ypixel)
	X,Y=np.meshgrid(shiftx,shifty)
	Z=beamprofileexposure_rot_hd(power,radiusx=X,radiusy=Y,wx=wx,wy=wy,mux=mux,muy=muy,reprate=reprate,theta=theta,radians=radians,materialreflectivity=materialreflectivity,fitp1=fitp1)
	if save:
		np.savetxt(destination,Z)
	return Z

def airy2dellipse(A,x=0,y=0.,wx=84./2.,wy=49./2,mux=0,muy=0,radians=0):
	return A*4*(special.jv(1,(((x+0.00000000000001)*2*2**.5/wx)**2+((y+0.00000000000001)*2*2**.5/wy)**2)**.5)**2/(((x+0.00000000000001)*2*2**.5/wx)**2+((y+0.00000000000001)*2*2**.5/wy)**2))

def airy2dellipse_rot(A,x=0,y=0.,wx=84./2.,wy=49./2,mux=0,muy=0,theta=0,radians=0):
	"""xp=(np.cos(theta)*(x-mux)-np.sin(theta)*(y-muy))
		yp=(np.sin(theta)*(x-mux)+np.cos(theta)*(y-muy))"""
	if(radians==0):
		theta=theta*3.1415259/180
	return A*4*(special.jv(1,((((np.cos(theta)*(x-mux)-np.sin(theta)*(y-muy))+0.00000000000001)*2*2**.5/wx)**2+(((np.sin(theta)*(x-mux)+np.cos(theta)*(y-muy))+0.00000000000001)*2*2**.5/wy)**2)**.5)**2/((((np.cos(theta)*(x-mux)-np.sin(theta)*(y-muy))+0.00000000000001)*2*2**.5/wx)**2+(((np.sin(theta)*(x-mux)+np.cos(theta)*(y-muy))+0.00000000000001)*2*2**.5/wy)**2))

def fraunhofer2dellipse_rot(amplitudes,orders,x=0,y=0.,wx=84./2.,wy=49./2,mux=0,muy=0,theta=0,radians=0):
	"""xp=(np.cos(theta)*(x-mux)-np.sin(theta)*(y-muy))
		yp=(np.sin(theta)*(x-mux)+np.cos(theta)*(y-muy))"""
	if(radians==0):
		theta=theta*3.1415259/180
		diffraction=0
	for i in range(len(amplitudes)):
		diffraction+=amplitudes[i]*4*(special.jv(orders[i],((((np.cos(theta)*(x-mux)-np.sin(theta)*(y-muy))+0.00000000000001)*2*2**.5/wx)**2+(((np.sin(theta)*(x-mux)+np.cos(theta)*(y-muy))+0.00000000000001)*2*2**.5/wy)**2)**.5)**2/((((np.cos(theta)*(x-mux)-np.sin(theta)*(y-muy))+0.00000000000001)*2*2**.5/wx)**2+(((np.sin(theta)*(x-mux)+np.cos(theta)*(y-muy))+0.00000000000001)*2*2**.5/wy)**2))
	return diffraction

def getfraunhoferprofilemat(power,amplitudes,orders,shiftx,shifty,wx,wy,mux=0,muy=0,l=0.515,depth=.816,reprate=1e6,xpixel=738,ypixel=485,theta=0,radians=0, save=0,destination="beamprofile.txt"):
	if(radians==1):
		theta=theta*3.1415259/180
	pixelvolume=(shiftx[1]-shiftx[0])*(shifty[1]-shifty[0])*depth*10**-12
	X,Y=np.meshgrid(shiftx,shifty)
	photons=energyinchamber(power,materialreflectivity=0.0,reprate=reprate)/energyofphoton(l)
	print("theta:",theta)
	Z=fraunhofer2dellipse_rot(amplitudes,orders,x=X,y=Y,wx=wx,wy=wy,mux=mux,muy=muy,theta=theta,radians=radians)
	Z_int=np.sum(Z)
	print(Z_int)
	profilemat=np.multiply(Z,photons/Z_int/pixelvolume/10**15)
	if save:
		np.savetxt(destination,Z)
	return profilemat

def getairyprofilemat(power,shiftx,shifty,wx,wy,mux=0,muy=0,l=0.515,depth=.816,reprate=1e6,xpixel=738,ypixel=485,theta=0,radians=0, save=0,destination="beamprofile.txt"):
	if(radians==1):
		theta=theta*3.1415259/180
	pixelvolume=(shiftx[1]-shiftx[0])*(shifty[1]-shifty[0])*depth*10**-12
	X,Y=np.meshgrid(shiftx,shifty)
	photons=energyinchamber(power,materialreflectivity=0.0,reprate=reprate)/energyofphoton(l)
	print("theta:",theta)
	Z=airy2dellipse_rot(1,x=X,y=Y,wx=wx,wy=wy,mux=mux,muy=muy,theta=theta,radians=radians)
	Z_int=np.sum(Z)
	print(Z_int)
	profilemat=np.multiply(Z,photons/Z_int/pixelvolume/10**15)
	if save:
		np.savetxt(destination,Z)
	return profilemat


def getprofilemat(power,mag,wx,wy,materialreflectivity=0.0,mux=0,muy=0,l=0.515,depth=.816,reprate=1e6,xpixel=738,ypixel=485,theta=0,radians=0, save=0,destination="beamprofile.txt"):
	print(theta)
	shiftx,shifty=getpositionaxis(mag, xpixel=xpixel,ypixel=ypixel)
	X,Y=np.meshgrid(shiftx,shifty)
	Z=beamprofilefluence_rot(power,radiusx=X,radiusy=Y,wx=wx,wy=wy,mux=mux,muy=muy,l=l,depth=depth,reprate=reprate,theta=theta,radians=radians)
	if save:
		np.savetxt(destination,Z)
	return Z

def getexposuremat(power,mag,wx,wy,materialreflectivity=0.0,mux=0,muy=0,reprate=1e6,xpixel=738,ypixel=485,theta=0,radians=0, save=0,destination="beamprofile.txt"):
	print(theta)
	shiftx,shifty=getpositionaxis(mag, xpixel=xpixel,ypixel=ypixel)
	X,Y=np.meshgrid(shiftx,shifty)
	Z=beamprofileexposure_rot(power,radiusx=X,radiusy=Y,wx=wx,wy=wy,mux=mux,muy=muy,reprate=reprate,theta=theta,radians=radians,materialreflectivity=materialreflectivity)
	if save:
		np.savetxt(destination,Z)
	return Z

def simulatepowerseries(powers,a,rho,mag,wx,wy,mux=0,muy=0,l=0.515,depth=.816,reprate=1e6,xpixel=738,ypixel=485,theta=0,radians=0, save=0,prefix="suemsim",verbose=0):
	"""
	module for simulating the sample SUEM response.
	all inputs are in units of microns and milliwats.
	:param float: power: unit mW power before the chamber
	:param float: radiusx: unit um position in beam x offset from center position
	:param float: radiusy: unit um position in beam y offset from center position
	:param float: wx: unit um 1/e2 radius of beam in x
	:param float: wy: unit um 1/e2 radius of beam in y
	:param float: reprate: unit hertz rep rate of pump. Note could be 2e6
	:param float: mux: unit um center position in beam x
	:param float: muy: unit um center position in beam y
	:param float: l: unit um wavelength of light used
	:param float: depth: unit um attenuation depth of light in the sample
	Example Code:
		if 1:
		ps.simulatepowerseries(pp,.182,450,mag0,84/2*1.2,49/2*1.2,mux=-30,muy=15,l=0.515,depth=.816,reprate=1e6,xpixel=738,ypixel=485,theta=0, save=1,prefix="suemsim",verbose=1)
	if 1:
		fl_simps=ps.getfilenames("suemsim",index=pp1,suffix='.txt',delimiter="_pump",verbose=1)
		sf_simps=ps.getstretchfactormulti(fl_simps,minimalstretch=1)
		ps.stretchdatamulti(fl_simps,sf_simps)
	if 1:
		fl_simps_stretched=ps.getfilenames("suemsim",index=pp1,suffix='_stretched.txt',delimiter="_pump",verbose=1)
		ps.adjustdatamulti(fl_simps_stretched,.46,.5)
		fl_simps_adj=ps.getfilenames("suemsim",index=pp1,suffix='_stretched_adj.txt',delimiter="_pump",verbose=1)
		ps.createimagemulti(fl_simps_adj)
	"""
	if(radians==0):
		theta=theta*3.1415259/180
	shiftx,shifty=getpositionaxis(mag, xpixel=xpixel,ypixel=ypixel)
	X,Y=np.meshgrid(shiftx,shifty)
	destinationtxt=""
	for p in powers:
		destinationtxt=prefix+"_pump"+formatdecimalstring(p,6)+".txt"
		if verbose:
			print(destinationtxt)
		Z=beamprofilefluence_rot(p,radiusx=X,radiusy=Y,wx=wx,wy=wy,mux=mux,muy=muy,l=l,depth=depth,reprate=reprate,theta=theta)
		suemsim=fluenceresponse0(a,rho,Z)
		if save:
			np.savetxt(destinationtxt,suemsim)
	return



#________________________________________________________1D Spatial Profiles_______________________________________________________________

def cumulative(x,a,mu,sigma,offset):
	#http://mathworld.wolfram.com/GaussianFunction.html
	#w is the firtting parameters of order 
	#w[0]=A
	#w[1]=mu
	#w[2]=sigmax
	return a*1/2*(1+special.erf((x-mu)/(sigma*2**.5)))+offset

def dgaussdx(x,a,mu,sigma,offset):
	#http://mathworld.wolfram.com/GaussianFunction.html
	#w is the firtting parameters of order 
	#w[0]=A
	#w[1]=mu
	#w[2]=sigmax
	return a*(x-mu)*np.exp(-((x-mu)**2/(2*sigma**2)))+offset

def cumulativeplusdgaussdx(x,a,arel,mu,sigma,offset):
	#http://mathworld.wolfram.com/GaussianFunction.html
	#w is the firtting parameters of order 
	#w[0]=A
	#w[1]=mu
	#w[2]=sigmax
	return dgaussdx(x,a*arel,mu,sigma,0)+cumulative(x,a,mu,sigma,offset)

def gaussplusdgaussdx(x,a,b,mu,sigma,offset):
	#http://mathworld.wolfram.com/GaussianFunction.html
	#w is the firtting parameters of order 
	#w[0]=A
	#w[1]=mu
	#w[2]=sigmax
	return (a+b*(x-mu))*np.exp(-((x-mu)**2/(2*sigma**2)))+offset

def supergaussplusline(x,a,b,amp,mu,wx,p):
	return a*(x)+b+amp*np.exp(-2*((x-mu)**2/(wx**2))**p)

def supergauss(x,amp,mu,wx,p):
	return amp*np.exp(-2*((x-mu)**2/(wx**2))**p)

def line(x,a,b):
	return a*(x)+b

def logesat(f,a,esat):
	return a*np.log(1+f/esat)

def expesat(f,a,esat):
	return a*(1-np.exp(-f/esat))

def loggaussplusline(x,a,b,amp,mu,wx,esat):
	return a*x+b+amp*np.log(1+np.exp(-2*((x-mu)**2/(wx**2)))/esat)

def loggauss(x,amp,mu,wx,esat):
	return amp*np.log(1+np.exp(-2*((x-mu)**2/(wx**2)))/esat)

def loggausspluslineheld(x,a,b,amp,esat):
	return a*x+b+amp*np.log(1+np.exp(-2*((x-6.15)**2/(42**2)))/esat)

def expgaussplusline(x,a,b,amp,mu,wx,esat):
	return a*x+b+amp*(1-np.exp(-np.exp(-2*((x-mu)**2/(wx**2)))/esat))

def expgauss(x,amp,mu,wx,esat):
	return amp*(1-np.exp(-np.exp(-2*((x-mu)**2/(wx**2)))/esat))

def expgaussheld2(x,amp,esat):
	scale=1
	return amp*(1-np.exp(-2*energyinchamber(50.,materialreflectivity=.31,reprate=2e6)/esat/np.pi/(scale**2)/24.5e-4/42e-4*np.exp(-2*((x-12)**2/(scale*42**2)))))

def expgausspluslineheld(x,a,b,amp,esat):
	return a*x+b+amp*(1-np.exp(-np.exp(-2*((x-6.15)**2/(42**2)))/esat))

def expgausspluslineheld2(x,a,b,amp,esat):
	scale=1
	return a*x+b+amp*(1-np.exp(-2*energyinchamber(50.,materialreflectivity=.31,reprate=2e6)/esat/np.pi/(scale**2)/24.5e-4/42e-4*np.exp(-2*((x-12)**2/(scale*42**2)))))

def expgausspluslineheld3(x,a,b,amp,kappa):
	return a*x+b+amp*(1-np.exp(-kappa*np.exp(-2*((x-33)**2/(42**2)))))

def expgaussplusline3(x,a,b,amp,mu,wx,kappa):
	return a*x+b+amp*(1-np.exp(-kappa*np.exp(-2*((x-mu)**2/(wx**2)))))

def expgauss3(x,amp,mu,wx,kappa):
	return amp*(1-np.exp(-kappa*np.exp(-2*((x-mu)**2/(wx**2)))))

def calcesat(kappa,P,R=0.31,omegarep=2e6,wx=42,wy=24.5):
	rho=2*(1-R)*P/kappa/np.pi/(wx*10**-4)/(wx*10**-4)/omegarep
	return rho

def trimData(data,timepoints,st):
	desiredLoc=(int(st)-int(timepoints[0]))/(int(timepoints[1])-int(timepoints[0]))
	dataTrim=np.delete(data,np.arange(desiredLoc),axis=1)
	timeTrim=np.delete(timepoints,np.arange(desiredLoc))
	print('The data has been trimmed at '+str(st)+ ' fs. timepoints has been shortened from '+str(len(timepoints))+' points to '+str(len(timeTrim))+' points.')
	return dataTrim,timeTrim

def createbounds(p0,boundfactor=2.0):
	lowerbound=[]
	upperbound=[]
	for p in p0:
		if p<0:
			lowerbound.append((p*boundfactor))
			upperbound.append((p/boundfactor))
		elif p>0:
			lowerbound.append((p/boundfactor))
			upperbound.append((p*boundfactor))
		elif p==0:
			lowerbound.append(-(1-boundfactor))
			upperbound.append(1-boundfactor)
	return (lowerbound,upperbound)

def fit2dring(filelist,p0,mag,xpixel=738,ypixel=485,verbose=0,**keyword_parameters):
	if ('boundfactor' in keyword_parameters):
		boundfactor = keyword_parameters['boundfactor']
	else:
		boundfactor=2.0
	if ('bounds' in keyword_parameters):
		bounds = keyword_parameters['bounds']
	else:
		bounds=createbounds(p0,boundfactor=boundfactor)
	if ('save' in keyword_parameters):
		save = keyword_parameters['save']
	else:
		save=1
	optparams=np.array([], dtype=float).reshape(0,len(p0))
	shiftx,shifty=getpositionaxis(mag, xpixel=xpixel,ypixel=ypixel)
	shiftx0,shifty0=np.meshgrid(shiftx,shifty)
	for f in filelist:
		data =np.loadtxt(f).ravel()
		if verbose:
			print("Prosessing :", f)
			start_time = time.time()
			print("initial guess: ", p0)
			print("lower bounds: ", bounds[0])
			print("upper bounds: ", bounds[1])
		popt, pcov = opt.curve_fit(multisuperguass2drotoffset4,(shiftx0,shifty0), data, p0=p0,bounds=bounds)
		if verbose:
			print("--- %s (s) to fit image --- " % (time.time() - start_time))
			print("Optimization Parameters: ",popt)
		fn=generatefilename(f,suffix="_fit",extension="txt",verbose=verbose)
		fitdata=multisuperguass2drotoffset4((shiftx0,shifty0),popt[0],popt[1],popt[2],popt[3],popt[4],popt[5],popt[6],popt[7]).reshape(485,738)
		np.savetxt(fn,fitdata)
		if save:
			im=Image.fromarray(np.uint8(fitdata*255))
			fntif=generatefilename(f,suffix="_fit",extension="tif",verbose=verbose)
			im.save(fntif)
		#createimage(fn,destination=fn)
		optparams=np.vstack((optparams,popt))
	fnopt=getfileprefix(filelist[0],delimiter="_")+"_popt.txt"
	np.savetxt(fnopt,optparams)
	return

#________________________________________________________2D Spatial Profiles_______________________________________________________________


def simulate2Dfunction(w,func='gauss2d',xpixel=738, ypixel=485,**keyword_parameters):
	"""
	Example Code:
	for i in range(20):
		print(i)
		sgauss=ps.simulate2Dfunction([1,366,84,242.5,49,.01*2**i,0],func='supergauss2drot',xpixel=738, ypixel=485)
		ps.createimage(sgauss,save=1,show=1,destination="supergauss"+str(i),filetype='tif')
	"""
	if ('mag' in keyword_parameters):
		mag = keyword_parameters['mag']
		shiftx,shifty=getpositionaxis(mag)
	else:
		shiftx=np.arange(xpixel)
		shifty=np.arange(ypixel)
	xv,yv = np.meshgrid(shiftx,shifty)
	if(func=='gauss2d'):
		z= gauss2d(xv,yv,w)
	elif(func=='gauss2drot'):
		z= gauss2drot(xv,yv,w)
	elif(func=='doublegauss2drot'):
		z= doublegauss2drot(xv,yv,w)
	elif(func=='supergauss2drot'):
		z= superguass2drot(xv,yv,w)
	elif(func=='doublesupergauss2drot'):
		z= doublesuperguass2drot(xv,yv,w)
	elif(func=='superguass2drotoffset'):
		z= superguass2drotoffset(xv,yv,w)
	elif(func=='doublesuperguass2drotoffset'):
		z = doublesuperguass2drotoffset(xv,yv,w)
	elif(func=='doublesuperguass2drot'):
		z = doublesuperguass2drot(xv,yv,w)
	elif(func=='multisuperguass2drotoffset'):
		z = multisuperguass2drotoffset(xv,yv,w)
	return z


def gauss2d(x,y,w):
	#http://mathworld.wolfram.com/GaussianFunction.html
	#w is the firtting parameters of order 
	#w[0]=A
	#w[1]=muy
	#w[2]=sigmay
	#w[3]=mux
	#w[4]=sigmax
	#return w[0]/(2*3.1415259*w[2]*w[4])*np.exp(-((y-w[1])**2/(2*w[2]**2)+(x-w[3])**2/(2*w[4]**2)))
	return w[0]*np.exp(-((y-w[1])**2/(2*w[2]**2)+(x-w[3])**2/(2*w[4]**2)))

def gauss2drot(x,y,w):
	#http://mathworld.wolfram.com/GaussianFunction.html
	#w is the firtting parameters of order 
	#w[0]=A
	#w[1]=mux
	#w[2]=sigmax
	#w[3]=muy
	#w[4]=sigmay
	#w[5]=theta
	#return w[0]/(2*3.1415259*w[2]*w[4])*np.exp(-(((np.cos(w[5])*x-np.sin(w[5])*y)-w[1])**2/(2*w[2]**2)+((np.cos(w[5])*x+np.sin(w[5])*y)-w[3])**2/(2*w[4]**2)))
	return w[0]*np.exp(-((np.cos(w[5])*(x-w[1])-np.sin(w[5])*(y-w[3]))**2/(2*w[2]**2)+(np.sin(w[5])*(x-w[1])+np.cos(w[5])*(y-w[3]))**2/(2*w[4]**2)))

def superguass2drot(x,y,w):
	#http://mathworld.wolfram.com/GaussianFunction.html
	#w is the firtting parameters of order 
	#w[0]=A
	#w[1]=mux
	#w[2]=sigmax
	#w[3]=muy
	#w[4]=sigmay
	#w[5]=Py
	#w[6]=theta
	#return w[0]/(2*3.1415259*w[2]*w[4])*np.exp(-(((np.cos(w[5])*x-np.sin(w[5])*y)-w[1])**2/(2*w[2]**2)+((np.cos(w[5])*x+np.sin(w[5])*y)-w[3])**2/(2*w[4]**2)))
	return w[0]*np.exp(-(((np.cos(w[6])*(x-w[1])-np.sin(w[6])*(y-w[3]))**2/(2*w[2]**2))+((np.sin(w[6])*(x-w[1])+np.cos(w[6])*(y-w[3]))**2/(2*w[4]**2)))**w[5])

def superguass2drotoffset(x,y,w):
	return superguass2drot(x,y,w[0:7])+w[7]

def doublesuperguass2drot(x,y,w):
	#w[0]=A1
	#w[1]=mux1
	#w[2]=sigmax1
	#w[4]=muy1
	#w[5]=sigmay1
	#w[6]=theta1
	#w[7]=Py1

	#w[0]=A2
	#w[8]=mux2
	#w[9]=sigmax2
	#w[11]=muy2
	#w[12]=sigmay2
	#w[13]=theta2
	#w[14]=Py2
	return superguass2drot(x,y,w[0:7])+superguass2drot(x,y,w[7:14])

def doublesuperguass2drotoffset(x,y,w):
	#w[0]=A1
	#w[1]=mux
	#w[2]=sigmax1
	#w[4]=muy1
	#w[5]=sigmay1
	#w[6]=theta1
	#w[7]=Py1
	#w[8]=A2
	#w[8]=mux
	#w[9]=sigmax2
	#w[11]=muy
	#w[12]=sigmay2
	#w[13]=theta2
	#w[14]=Py2
	#w[15]=zoffset
	return doublesuperguass2drot(x,y,w[0:14])+w[14]

def multisuperguass2drotoffset(x,y,w):
	#same centercoord and same theta
	#w[0]=Ai
	#w[1]=sigmaxi
	#w[2]=sigmayi
	#w[3]=Pi
	#w[4]=mux
	#w[5]=muy
	#w[6]=theta
	#w[7]=zoffset
	val=w[-1]
	for i in range(int((len(w)-4.0)/4.0)):
		val+=w[4*i+0]*np.exp(-(((np.cos(w[-2])*(x-w[-4])-np.sin(w[-2])*(y-w[-3]))**2/(2*w[4*i+1]**2))+((np.sin(w[-2])*(x-w[-4])+np.cos(w[-2])*(y-w[3]))**2/(2*w[4*i+2]**2)))**w[4*i+3])
	return val


def doublegauss2drot(x,y,w):
	#http://mathworld.wolfram.com/GaussianFunction.html
	#w is the firtting parameters of order 
	#w[0]=A1
	#w[1]=A2
	#w[2]=mux
	#w[3]=muy
	#w[4]=sigmax1
	#w[5]=sigmay1
	#w[6]=sigmax2
	#w[7]=sigmay2
	#w[8]=theta
	#w[9]=zoffset
	#w[0]=A1, w[1]=A2 , w[2]=mux , w[3]=muy , w[4]=sigmax1 , w[5]=sigmay1 , w[6]=sigmax2 , w[7]=sigmay2 , w[8]=theta , ,w[9]=zoffset , 
	#w[1]=A1*np.exp(-((np.cos(theta)*(x-mux)-np.sin(theta)*(y-muy))**2/(2*sigmax1**2)+(np.sin(theta)*(x-mux)+np.cos(theta)*(y-muy))**2/(2*sigmay1**2)))+A2*np.exp(-((np.cos(theta)*(x-mux)-np.sin(theta)*(y-muy))**2/(2*sigmax2**2)+(np.sin(theta)*(x-mux)+np.cos(theta)*(y-muy))**2/(2*wsigmay2**2)))+zoffset
	return w[0]*np.exp(-((np.cos(w[8])*(x-w[2])-np.sin(w[8])*(y-w[3]))**2/(2*w[4]**2)+(np.sin(w[8])*(x-w[2])+np.cos(w[8])*(y-w[3]))**2/(2*w[5]**2)))+w[1]*np.exp(-((np.cos(w[8])*(x-w[2])-np.sin(w[8])*(y-w[3]))**2/(2*w[6]**2)+(np.sin(w[8])*(x-w[2])+np.cos(w[8])*(y-w[3]))**2/(2*w[7]**2)))+w[9]

def twoD_Gaussian(xymesh, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
	xo = float(xo)
	yo = float(yo)	
	a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
	b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
	c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
	g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
							+ c*((y-yo)**2)))
	return g.ravel()

def multisuperguass2drotoffset(xymesh,w):
	#same centercoord and same theta
	#w[0]=Ai
	#w[1]=sigmaxi
	#w[2]=sigmayi
	#w[3]=Pi
	#w[4]=mux
	#w[5]=muy
	#w[6]=theta
	#w[7]=zoffset
	x=xymesh[0]
	y=xymesh[1]
	g=np.ones(np.shape(x))*w[-1]
	for i in range(int((len(w)-4.0)/4.0)):
		g+=w[4*i+0]*np.exp(-(((np.cos(w[-2])*(x-w[-4])-np.sin(w[-2])*(y-w[-3]))**2/(2*w[4*i+1]**2))+((np.sin(w[-2])*(x-w[-4])+np.cos(w[-2])*(y-w[3]))**2/(2*w[4*i+2]**2)))**w[4*i+3])
	return g.ravel()

def multisuperguass2drotoffset2(xymesh,w0,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11):
	#same centercoord and same theta
	#w[0]=Ai
	#w[1]=sigmaxi
	#w[2]=sigmayi
	#w[3]=Pi
	#w[4]=mux
	#w[5]=muy
	#w[6]=theta
	#w[7]=zoffset
	x=xymesh[0]
	y=xymesh[1]
	g=np.ones(np.shape(x))*w11
	g+=w0*np.exp(-(((np.cos(w10)*(x-w8)-np.sin(w10)*(y-w9))**2/(2*w1**2))+((np.sin(w10)*(x-w8)+np.cos(w10)*(y-w9))**2/(2*w2**2)))**w3)
	g+=w4*np.exp(-(((np.cos(w10)*(x-w8)-np.sin(w10)*(y-w9))**2/(2*w5**2))+((np.sin(w10)*(x-w8)+np.cos(w10)*(y-w9))**2/(2*w6**2)))**w7)
	return g.ravel()

def multisuperguass2drotoffset3(xymesh,w0,w1,w2,w3,w4,w5,w6,w7,w8,w9):
	#same centercoord and same theta
	#w[0]=Ai
	#w[1]=sigmaxi
	#w[2]=Pi
	#w[3]=mux
	#w[4]=muy
	#w[5]=theta
	#w[6]=zoffset 
	x=xymesh[0]
	y=xymesh[1]
	g=np.ones(np.shape(x))*w9
	g+=w0*np.exp(-(((np.cos(w8)*(x-w6)-np.sin(w8)*(y-w7))**2/(2*w1*1.71429**2))+((np.sin(w8)*(x-w6)+np.cos(w8)*(y-w7))**2/(2*w1**2)))**w2)
	g+=w3*np.exp(-(((np.cos(w8)*(x-w6)-np.sin(w8)*(y-w7))**2/(2*w4*1.71429**2))+((np.sin(w8)*(x-w6)+np.cos(w8)*(y-w7))**2/(2*w4**2)))**w5)
	return g.ravel()


def multisuperguass2drotoffset4(xymesh,w0,w1,w2,w3,w4,w5,w6,w7):
	#same centercoord and same theta
	#w[0]=Ai
	#w[1]=sigmaxi
	#w[2]=Pi
	#w[3]=mux
	#w[4]=muy
	#w[5]=theta
	#w[6]=zoffset
	x=xymesh[0]
	y=xymesh[1]
	g=np.ones(np.shape(x))*w7
	g-=w0*np.exp(-(((np.subtract(x,w5))**2/(2*(w1*1.71429)**2))+(np.subtract(y,w6))**2/(2*w1**2))**w2)
	g+=w0*np.exp(-(((np.subtract(x,w5))**2/(2*(w3*1.71429)**2))+(np.subtract(y,w6))**2/(2*w3**2))**w4)
	return g.ravel()
#________________________________________________________Power/Fluence Profiles_______________________________________________________________


"""
========================================================================================================
SECTION 99 useful functions for image data analysis
========================================================================================================
"""
def RoundToSigFigs_fp( x, sigfigs ):
    """
    Rounds the value(s) in x to the number of significant figures in sigfigs.
    Return value has the same type as x.

    Restrictions:
    sigfigs must be an integer type and store a positive value.
    x must be a real value.
    """
    __logBase10of2 = 3.010299956639811952137388947244930267681898814621085413104274611e-1
    if not ( type(sigfigs) is int or type(sigfigs) is long or
             isinstance(sigfigs, np.integer) ):
        raise TypeError( "RoundToSigFigs_fp: sigfigs must be an integer." )

    if sigfigs <= 0:
        raise ValueError( "RoundToSigFigs_fp: sigfigs must be positive." )

    if not np.isreal( x ):
        raise TypeError( "RoundToSigFigs_fp: x must be real." )

    xsgn = np.sign(x)
    absx = xsgn * x
    mantissa, binaryExponent = np.frexp( absx )

    decimalExponent = __logBase10of2 * binaryExponent
    omag = np.floor(decimalExponent)

    mantissa *= 10.0**(decimalExponent - omag)

    if mantissa < 1.0:
        mantissa *= 10.0
        omag -= 1.0

    return xsgn * np.around( mantissa, decimals=sigfigs - 1 ) * 10.0**omag

def dot(x, y):
    """Dot product as sum of list comprehension doing element-wise multiplication"""
    return sum(x_i*y_i for x_i, y_i in zip(x, y))

def lin_equ(l1, l2):
	"""
	paramsdtupleLine encoded as l=(x,y)."""
	m = Decimal((l2[1] - l1[1])) / Decimal(l2[0] - l1[0])
	c = (l2[1] - (m * l2[0]))
	return m, c

def lin_coords(coords):
	lin_coefs=np.array([], dtype=np.int64).reshape(0,2)
	for i in range(coords):
		m,c=lin_equ(coords[:,i], coords[:,i+1])
		np.vstack((lin_coefs,np.array([m,c])))
	return lin_coefs


def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def vectorize(line):
	return [line[1,0]-line[0,0],line[1,1]-line[0,1]]


def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))


def length(v):
  return math.sqrt(dotproduct(v, v))

def image2array(im):
	imarray = np.array(im)
	return imarray

def array2image(array):
	im=Image.fromarray(np.uint8(cm.gist_earth(array)*255))


def grayscaleimage(im):
	im_RGB = im.convert("RGB")
	#Mae it an array
	imarray_RGB = np.array(im_RGB)
	#convert the array to grayscale
	imarray_gray=np.dot(imarray_RGB, [0.2989, 0.5870, 0.1140])
	return imarray_gray


def line_intersection(line1, line2):
	xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
	ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

	def det(a, b):
		return a[0] * b[1] - a[1] * b[0]

	div = det(xdiff, ydiff)
	if div == 0:
	   raise Exception('lines do not intersect')

	d = (det(*line1), det(*line2))
	x = det(d, xdiff) / div
	y = det(d, ydiff) / div
	return x, y

def rotateimage(im,hline,vline):

	#vectorize the lines that were drawn
	hvector=vectorize(np.array(hline))
	vvector=vectorize(np.array(vline))
	#define true horizontal and vertical vectors
	vho=[1,0]
	vvert=[0,-1]
	#get the angle between lines and vectors
	theta0=angle(hvector,vho)*180/3.1415
	theta1=angle(vvector,vvert)*180/3.1415
	#take an average
	theta_avg=(theta0+theta1)/2
	if 1:
		print("Rotating image by ", -theta_avg , "o")
	
	#Load the image with PIL
	#rotate the image
	imrot=im.rotate(-theta_avg)
	return imrot

def translateimage(im, hline, vline):
	#get the intersection of the horizontal and vertical lines
	xline=line_intersection(hline, vline)
	#get the index of the middle pixels of the image as [x,y]
	midim=np.flip(np.divide(np.asarray(np.shape(im)[0:2]),2))
	#get the offset of the intersection from the middle
	delmid=np.subtract(xline,midim)
	if 1:
		print("Translating image by ", delmid , "pixels")
	
	#Convert the Image to RGB a mxnx3 dimensional array where three is the colors
	imtrans=im.transform(im.size, Image.AFFINE, (1,0,delmid[0],0,1,delmid[1]))
	return imtrans

def removeinstrumentresponse(imarray, irfarray, scale=1):
	#we can't divide by zero so lets set the minimum value of irf to the median/10
	irffloored = np.add(irfarray,np.median(irfarray)/10)
	#was the irf measured with the same beam current ie. 
	#Apature? if no you should divide your irf by scale 
	#where scale is the ratio of beam current of irf/beam current of plume.
	irfscaled= np.divide(irffloored,scale)
	#first subtract the back ground emission. Then divide by the phosphor responsivity.
	imadj=np.divide(np.subtract(imarray,irfscaled),irfscaled)
	imadj2=np.divide(np.add(imadj,np.amin(imadj)),np.amax(imadj)-np.amin(imadj))
	return imadj2

def clockwiseangle_and_distance(point):
	vector = [point[0]-origin[0], point[1]-origin[1]]
	lenvector = math.hypot(vector[0], vector[1])
	if lenvector == 0:
		return -math.pi, 0
	normalized = [vector[0]/lenvector, vector[1]/lenvector]
	dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]
	diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]
	angle = math.atan2(diffprod, dotprod)
	if angle < 0:
		return 2*math.pi+angle, lenvector
	return angle, lenvector

# shoelace formula implementation
def PolygonArea(corners):
	n = len(corners)
	area = 0.0
	for i in range(n):
		j = (i + 1) % n
		area += corners[i][0] * corners[j][1]
		area -= corners[j][0] * corners[i][1]
	area = abs(area) / 2.0
	return area



def applylowpassfilter(filename,show=0):
	image = cv.imread(filename)
	image2 = cv.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#figure_size = 9
	dft = cv2.dft(np.float32(image2),flags = cv2.DFT_COMPLEX_OUTPUT)
	# shift the zero-frequncy component to the center of the spectrum
	dft_shift = np.fft.fftshift(dft)
	# save image of the image in the fourier domain.
	#magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
	# plot both images
	#plt.figure(figsize=(11,6))
	#plt.subplot(121),plt.imshow(image2, cmap = 'gray')
	#plt.title('Input Image'), plt.xticks([]), plt.yticks([])
	#plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
	#plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
	#plt.show()
	rows, cols = image2.shape
	crow,ccol = rows//2 , cols//2
	lpf=130
	# create a mask first, center square is 1, remaining all zeros
	mask = np.zeros((rows,cols,2),np.uint8)
	mask[crow-lpf:crow+lpf, ccol-lpf:ccol+lpf] = 1
	# apply mask and inverse DFT
	fshift = dft_shift*mask
	f_ishift = np.fft.ifftshift(fshift)
	img_back = cv2.idft(f_ishift)
	img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])
	# plot both images
	if(show):
		plt.figure(figsize=(11,6))
		plt.subplot(121),plt.imshow(image2, cmap = 'gray')
		plt.title('Input Image'), plt.xticks([]), plt.yticks([])
		plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
		plt.title('Low Pass Filter'), plt.xticks([]), plt.yticks([])
		plt.show()
	return img_back


def points_in_circle(radius, x0=0, y0=0, ):
	x_ = np.arange(x0 - radius - 1, x0 + radius + 1, dtype=int)
	y_ = np.arange(y0 - radius - 1, y0 + radius + 1, dtype=int)
	x, y = np.where((x_[:,np.newaxis] - x0)**2 + (y_ - y0)**2 <= radius**2)
	# x, y = np.where((np.hypot((x_-x0)[:,np.newaxis], y_-y0)<= radius)) # alternative implementation
	for x, y in zip(x_[x], y_[y]):
		yield x, y

def highpass(shape):
	"""Return highpass filter to be multiplied with fourier transform."""
	x = np.outer(
		np.cos(np.linspace(-math.pi/2.0, math.pi/2.0, shape[0])),
		np.cos(np.linspace(-math.pi/2.0, math.pi/2.0, shape[1])))
	return (1.0 - x) * (2.0 - x)

def mean_confidence_interval(data, confidence=0.95):
	a = 1.0 * np.array(data)
	n = len(a)
	m, se = np.mean(a), scipy.stats.sem(a)
	h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
	return m, m-h, m+h