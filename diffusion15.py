import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import path
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
from matplotlib import rc, rcParams
from time import gmtime, strftime
import sys, time
import scipy
import scipy.sparse
import scipy.sparse.linalg
from scipy import integrate
import pysuem as ps
import pandas as pd



def diffpde_saturable_langmuir(hnu,nsat,De,K, tauinf,alpha,L, dt,T,F=0.49,verbose=1,returnminimal=0):
	"""
	u_n,rho_t,x,t,timeelasped =diffpde_saturable_perfect_sink(a, L, dt, F, T)
	Simplest expression of the computational algorithm
	using the Forward Euler method and explicit Python loops.
	For this method F <= 0.5 for stability.
	"""
	print("K,tauinf,alpha:",K, tauinf,alpha)
	t0 = time.time()
	Nt = int(np.round(T/float(dt)))
	timepoints = np.linspace(0, Nt*dt, Nt+1)   # Mesh points in time
	dydx=np.zeros(Nt+1)
	rho_t=np.zeros(Nt+1)
	rho_tot=0
	dxn = np.sqrt(De*dt/F)
	Nxn = int(round(L/dxn))
	print(Nxn)
	xn = np.linspace(0, L, Nxn+1)	   # Mesh points in space
	# Make sure dx and dt are compatible with x and t
	ue_removed=np.zeros(Nt+1)
	utxn=np.zeros((Nt+1,Nxn+1))
	dt = timepoints[1] - timepoints[0]
	ue   = np.zeros(Nxn+1)
	u_n   = np.zeros(Nxn+1)
	absdepth=.05 # um
	u_n=np.exp(-abs(xn-xn[round(Nxn/2)])/0.01)
	u_n=hnu*2*u_n/np.sum(u_n)
	for n in range(0, Nt):
		# Compute u at inner mesh points
		for i in range(1, Nxn):
			ue[i] = u_n[i] + F*(u_n[i-1] - 2*u_n[i] + u_n[i+1])
		# Insert boundary conditions
		ue[0] = 0;  ue[Nxn] = 0;
		#first calculate how many electrons have been lost to recombination
		ue_removed[n]=ue_removed[n-1]+rho_t[n-1]*dt/tauinf*(1+alpha*rho_t[n-1])
		rho_t[n]=rho_t[n-1]-rho_t[n-1]*dt/tauinf*(1+alpha*rho_t[n-1])
		#next take to total number of electrons rhot_t+un[round(Nxn/2)] and calculate the new equilibrium value
		rho_tot=rho_t[n]+ue[round(Nxn/2)]
		rho_t[n]=(1+K*nsat+K*rho_tot-(-4*K**2*nsat*rho_tot+(-1-K*nsat-K*rho_tot)**2)**0.5)/(2*K)
		ue[round(Nxn/2)] = rho_tot-	rho_t[n]
		utxn[n,:]=ue
		# Switch variables before next step
		#u_n[:] = u  # safe, but slow
		u_n, ue = ue, u_n
	t1=time.time()
	rho_t=np.roll(rho_t, 1)
	ue_removed=np.roll(ue_removed, 1)
	if verbose:
		print("--- %s seconds to run simulation--- " % (t1 - t0))
	if returnminimal:
		return rho_t
	else:
		return utxn,ue_removed,u_n,rho_t, xn,timepoints, t1-t0#,flux  # u_n holds latest u
	return

def diffpde_saturable_statistical_electron_sink_thermionic_decay(hnu,nsat,De,alpha,gamma, tauinf,L, dt,T, F=0.49,sinktype="none",verbose=1,returnminimal=0):
	"""
	u_n,rho_t,x,t,timeelasped =diffpde_saturable_perfect_sink(a, L, dt, F, T)
	Simplest expression of the computational algorithm
	using the Forward Euler method and explicit Python loops.
	For this method F <= 0.5 for stability.
	"""
	t0 = time.time()
	#print("sink type:" , sinktype)
	Nt = int(np.round(T/float(dt)))
	timepoints = np.linspace(0, Nt*dt, Nt+1)   # Mesh points in time
	dydx=np.zeros(Nt+1)
	rho_t=np.zeros(Nt+1)

	dxn = np.sqrt(De*dt/F)
	Nxn = int(round(L/dxn))
	xn = np.linspace(0, L, Nxn+1)	   # Mesh points in space
	# Make sure dx and dt are compatible with x and t
	ue_removed=np.zeros(Nt+1)
	utxn=np.zeros((Nt+1,Nxn+1))
	dt = timepoints[1] - timepoints[0]
	ue   = np.zeros(Nxn+1)
	u_n   = np.zeros(Nxn+1)
	#u_n = np.zeros(Nx+1)

	# Set initial condition u(x,0) = I(x)
	# for p in range(0,Nxn+1):
	# 	u_p[p] = I(xp[p],hnu,xp[round(Nxp/2)],l=0.87)
	# for o in range(0,Nxn+1):
	# 	u_n[o] = I(xn[o],hnu,xn[round(Nxn/2)],l=0.87)
	u_n=np.exp(-abs(xn-xn[round(Nxn/2)])/1)
	u_n=hnu*2*u_n/np.sum(u_n)
	for n in range(0, Nt):
		# Compute u at inner mesh points
		for i in range(1, Nxn):
			ue[i] = u_n[i] + F*(u_n[i-1] - 2*u_n[i] + u_n[i+1])
		# Insert boundary conditions
		ue[0] = 0;  ue[Nxn] = 0;
		#dydx[n]=(u_n[round(Nx/2)+1]-u_n[round(Nx/2)])/(x[round(Nx/2)+1]-x[round(Nx/2)])#+(u[round(Nx/2)+0]-u[round(Nx/2)-1])/(x[round(Nx/2)+0]-x[round(Nx/2)-1])
		#dydx[n]=(u[round(Nx/2)+0]-u[round(Nx/2)-1])/(x[round(Nx/2)+0]-x[round(Nx/2)-1])
		#turn saturable statistical sink on off
		if sinktype=="statistical saturable":
			rho_t[n]=rho_t[n-1]+ue[round(Nxn/2)]*De*(1-rho_t[n-1]/nsat)-rho_t[n-1]*dt/tauinf*(1+alpha*rho_t[n-1])**gamma
			ue_removed[n]=ue_removed[n-1]+ue[round(Nxn/2)]*De*(1-rho_t[n-1]/nsat)
			#print("time",timepoints[n],(1-rho_t[n-1]/nsat),"adding to rho",rho_t[n-1],ue_removed[n],uh_removed[n])
			ue[round(Nxn/2)] = ue[round(Nxn/2)]*De*(rho_t[n-1]/nsat);
			#turn saturable perfect sink on
		elif sinktype=="perfect thermionic":
			rho_t[n]=rho_t[n-1]+ue[round(Nxn/2)]-rho_t[n-1]*dt/tauinf*(1+alpha*rho_t[n-1])**gamma
			ue_removed[n]=ue_removed[n-1]+ue[round(Nxn/2)]
			ue[round(Nxn/2)] = 0;
		elif sinktype=="perfect saturable":
			if rho_t[n-1]<nsat and rho_t[n-1]>=0:
				#print("adding to rho",rho_t[n-1], (rho_t[n-1]<nsat and rho_t[n-1]>0))
				rho_t[n]=rho_t[n-1]+ue[round(Nxn/2)]
				ue_removed[n]=ue_removed[n-1]+ue[round(Nxn/2)]
				ue[round(Nxn/2)] = 0;
			elif rho_t[n-1]<nsat:
				rho_t[n]=rho_t[n-1]+ue[round(Nxn/2)]
				ue_removed[n]=ue_removed[n-1]+ue[round(Nxn/2)]
				ue[round(Nxn/2)] = 0;
			elif rho_t[n-1]>=0:
				rho_t[n]=rho_t[n-1]-uh[round(Nxp/2)]
				ue_removed[n]=ue_removed[n-1]
				uh[round(Nxp/2)] = 0;
			else:
				rho_t[n]=rho_t[n-1]
				ue_removed[n]=ue_removed[n-1]
		elif sinktype=="perfect":
			rho_t[n]=rho_t[n-1]+ue[round(Nxn/2)]
			ue_removed[n]=ue_removed[n-1]+ue[round(Nxn/2)]
			ue[round(Nxn/2)] = 0;
		elif sinktype=="none":
			rho_t[n]=rho_t[n-1]
			ue_removed[n]=0
		utxn[n,:]=ue
		# Switch variables before next step
		#u_n[:] = u  # safe, but slow
		u_n, ue = ue, u_n
	t1=time.time()
	rho_t=np.roll(rho_t, 1)
	ue_removed=np.roll(ue_removed, 1)
	#flux=-dydx*De
	#flux=np.cumsum(dydx*De)
	#print(flux)
	if verbose:
		print("--- %s seconds to run simulation--- " % (t1 - t0))
	if returnminimal:
		return rho_t
	else:
		return utxn,ue_removed,u_n,rho_t, xn,timepoints, t1-t0#,flux  # u_n holds latest u

def simulaterhoseries(L,De,Dh,dt,maxT,F,rhoseries,nsat,destination,sinktype='perfect'):
	Nt = int(np.round(maxT/float(dt)))
	time = np.linspace(0, Nt*dt, Nt+1)
	rhotmat=np.array([], dtype=np.int64).reshape(0,len(time))
	for i in rhoseries:
		rhotmat=np.vstack((rhotmat,diffpde_saturable_statistical_sink(i,nsat,De,Dh, L, dt,  maxT,F,sinktype,verbose=0,returnminimal=1)))
	np.savetxt(destination,rhotmat,delimiter='\t', newline='\n', fmt='%1.18f')
	np.savetxt(destination[0:-4]+"_timepoints.txt",time)
	np.savetxt(destination[0:-4]+"_rho0points.txt",rhoseries)
	return time,rhoseries,rhotmat

def simulaterhoseries_thermionicdecay(L,De,alpha,gamma,tauinf,dt,maxT,F,rhoseries,nsat,destination,sinktype='statistical saturable'):
	Nt = int(np.round(maxT/float(dt)))
	time = np.linspace(0, Nt*dt, Nt+1)
	rhotmat=np.array([], dtype=np.int64).reshape(0,len(time))
	for i in rhoseries:
		rhotmat=np.vstack((rhotmat,diffpde_saturable_statistical_electron_sink_thermionic_decay(i,nsat,De,alpha,gamma,tauinf, L, dt,T, F,sinktype=sinktype,verbose=1,returnminimal=1)))
	np.savetxt(destination,np.transpose(rhotmat))
	np.savetxt(destination[0:-4]+"_timepoints.txt",time)
	np.savetxt(destination[0:-4]+"_rho0points.txt",rhoseries)
	return time,rhoseries,rhotmat

def simulaterhoseries_langmuir(L,De,K,tauinf,alpha,dt,maxT,F,rhoseries,nsat,destination,):
	Nt = int(np.round(maxT/float(dt)))
	time = np.linspace(0, Nt*dt, Nt+1)
	rhotmat=np.array([], dtype=np.int64).reshape(0,len(time))
	for i in rhoseries:
		rhotmat=np.vstack((rhotmat,diffpde_saturable_langmuir(i,nsat,De,K,tauinf,alpha, L, dt,T, F,verbose=1,returnminimal=1)))
	np.savetxt(destination,np.transpose(rhotmat))
	np.savetxt(destination[0:-4]+"_timepoints.txt",time)
	np.savetxt(destination[0:-4]+"_rho0points.txt",rhoseries)
	return time,rhoseries,rhotmat

def plotrhoseries(timepoint,rhopoints,rhotmat,xmin=-.111,xmax=0.111,destination="simperfectdiffpde0.tif"):
	rc('font', weight='bold')
	fig, ax = plt.subplots(figsize=(8.33,8.33))
	plt.subplots_adjust(left = 0.15,right = 0.97,bottom = 0.15,top = 0.94)
	colors = [plt.cm.nipy_spectral(i) for i in np.array([0.9,0.8,0.55,0.2,0.07,0])]
	ax.set_prop_cycle('color', colors)
	ax.xaxis.set_minor_locator(AutoMinorLocator())
	ax.yaxis.set_minor_locator(AutoMinorLocator())
	ax.xaxis.set_tick_params(labelsize=12)
	ax.yaxis.set_tick_params(labelsize=12)
	ax.tick_params(which='minor', length=2)
	ax.tick_params(which='major', length=5)
	for tick in ax.get_xticklabels():
		tick.set_fontname("Arial")
	for tick in ax.get_yticklabels():
		tick.set_fontname("Arial")
	for i in range(len(rhopoints)):
		label0=str(rhopoints[i]/0.95)[0:6]+" $\\times10^{12} cm^-2$"
		plt.plot(timepoint,rhotmat[i,:],label=label0)
	ax.set_xlabel(r'Time Delay (ns)',fontsize=12, fontweight='bold',fontname="Arial")
	ax.set_ylabel(r'Occupied Surface States $n_c$ $(\times10^{12})$',fontsize=12, fontweight='bold',fontname="Arial")
	plt.legend(loc='right',framealpha=0.3,fontsize=12)
	if xmin !=-0.111:
		plt.xlim([xmin,xmax])
	fig.savefig(destination)
	plt.show()
	return

def getrhotmat_thermionic(L,De,alpha,gamma,tauinf,dt,maxT,F,minrho,maxrho,Nrho,nsat,destination,sinktype='statistical saturable'):
	Nt = int(np.round(T/float(dt)))
	time = np.linspace(0, Nt*dt, Nt+1)
	rhopoints=np.geomspace(minrho,maxrho, Nrho)
	flupoints=rhopoints*10**12*(3.73657*10**-19)*10**6
	rhotmat=np.array([], dtype=np.int64).reshape(0,len(time))
	for i in rhopoints:
		rhotmat=np.vstack((rhotmat,diffpde_saturable_statistical_electron_sink_thermionic_decay(i,nsat,De,alpha,gamma,tauinf, L, dt,T, F,sinktype=sinktype,verbose=1,returnminimal=1)))
	np.savetxt(destination,np.vstack((time,rhotmat)),delimiter='\t', newline='\n', fmt='%1.18f')
	#np.savetxt(destination,np.transpose(rhotmat))
	np.savetxt(destination[0:-4]+"_timepoints.txt",time)
	np.savetxt(destination[0:-4]+"_rho0points.txt",rhopoints)
	np.savetxt(destination[0:-4]+"_flupoints.txt",flupoints)
	print(flupoints)
	return time,rhopoints,rhotmat


def getrhotmat(L,De,Dh,dt,maxT,F,minrho,maxrho,Nrho,nsat,destination,sinktype='perfect'):
	Nt = int(np.round(T/float(dt)))
	time = np.linspace(0, Nt*dt, Nt+1)
	rhopoints=np.geomspace(minrho,maxrho, Nrho)
	rhotmat=np.array([], dtype=np.int64).reshape(0,len(time))
	for i in rhopoints:
		rhotmat=np.vstack((rhotmat,diffpde_saturable_statistical_sink(i,nsat,De,Dh, L, dt,  T,F,sinktype,verbose=0,returnminimal=1)))
	np.savetxt(destination,rhotmat)
	np.savetxt(destination[0:-4]+"_timepoints.txt",time)
	np.savetxt(destination[0:-4]+"_rho0points.txt",rhopoints)
	return time,rhopoints,rhotmat

def getrhotmatlangmuir(L,De,Dh,dt,maxT,F,minrho,maxrho,Nrho,nsat,destination,sinktype='perfect'):
	Nt = int(np.round(T/float(dt)))
	time = np.linspace(0, Nt*dt, Nt+1)
	rhopoints=np.geomspace(minrho,maxrho, Nrho)
	rhotmat=np.array([], dtype=np.int64).reshape(0,len(time))
	for i in rhopoints:
		rhotmat=np.vstack((rhotmat,diffpde_saturable_statistical_sink(i,nsat,De,Dh, L, dt,  T,F,sinktype,verbose=0,returnminimal=1)))
	np.savetxt(destination,rhotmat)
	np.savetxt(destination[0:-4]+"_timepoints.txt",time)
	np.savetxt(destination[0:-4]+"_rho0points.txt",rhopoints)
	return time,rhopoints,rhotmat


def I(x,amp,mux,l=0.87):
	"""Gaussian profile as initial condition."""
	return 1/4/1.545*amp*np.exp(-abs(x-mux)/0.87)

def conv(timepoints,rho_t,tau_n1):
	t2=np.linspace(-timepoints[-1],timepoints[-1],2*len(timepoints)-1)
	print(2*len(timepoints)-1)
	rho_t2=np.append(np.zeros(len(timepoints)-1),rho_t)
	exp=1/tau_n1*np.exp(((t2)/(tau_n1/10**3)))*np.heaviside((-t2), 0)
	ssexp=np.convolve(rho_t, exp, mode="full")
	s2=ssexp[int(np.round(len(ssexp)*0)):int(np.round(len(ssexp)*2/3))]
	print(len(s2))
	return t2,s2

def convmulti(timepoints,rhotmat,tau_n1,save=1,destination="rhotmat_alphabeta.txt"):
	rhotmat_conv=np.array([], dtype=np.int64).reshape(0,2*len(timepoints)-1)
	for i in range(np.shape(rhotmat)[1]):
		t2,s2=conv(timepoints,rhotmat[:,i],tau_n1)
		np.savetxt(destination[0:-4]+"_rhot_conv"+str(i)+".txt",s2)
		rhotmat_conv=np.vstack((rhotmat_conv,s2))
	np.savetxt(destination[0:-4]+"_time_conv.txt",t2)
	np.savetxt(destination,np.transpose(rhotmat_conv))
	return t2,rhotmat_conv

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

L=20
De=20# um^2/ns or 0.1cm^2/s
#1um^2/ns =10 cm^2/s
#Dh=0.001 # um^2/ns
Dh=0 # um^2/ns
print("De: ", De, "Dh: ", Dh)
#x = np.linspace(0, L, Nx+1)
dt=.002
T=10
print("timesteps:",T/dt)
F=.49
hnu=6/0.73*1.8 #*10^12 carriers/cm2
nsat=20
destination="rhotmat0De0p601Dh0p2651_tsink0_401mesh.txt"
alpha=0
gamma=1.0
#alpha=1.4
#gamma=4.8
tauinf=5
K=.5
wx0=42
wy0=26
print("K:", K)
#time0,rhopoints0,rhotmat=getrhotmat(L,De,Dh,.01,8,F,.1,10,401,2,destination,sinktype='perfect')
#rhoseries=np.array([7.2,5.2,3.7,2.5])*(11/65)/(962/2100)*10**-6/((3*10**8)*(6.626176*10**-34)/(515*10**-9))/10**12
rhoseries=np.array([9.27,7.917,5.57, 3.591,1.91,.8])*10**-6/((3*10**8)*(6.626176*10**-34)/(532*10**-9))/10**12
if 0:
	time0,rhopoints0,rhotmat0=simulaterhoseries(L,De,Dh,.005,8,F,rhoseries,2,destination,sinktype='perfect')
	plotrhoseries(time0,rhopoints0,rhotmat0,destination="simperfectdiffpde0.tif")
if 0:
	time0,rhopoints0,rhotmat0=simulaterhoseries(L,De,Dh,.005,8,F,rhoseries,2,destination,sinktype='statistical saturable')
	plotrhoseries(time0,rhopoints0,rhotmat0,destination="simstatisticalsaturablediffpde0.tif")
if 0:
	time0,rhopoints0,rhotmat0=simulaterhoseries_thermionicdecay(L,De,alpha,gamma,tauinf,.001,8,F,rhoseries,nsat,destination,sinktype='statistical saturable')
	plotrhoseries(time0,rhopoints0,rhotmat0,destination="simstatisticalsaturablediffpde0.tif")
if 0:
	#getrhotmat_thermionic(L,De,alpha,gamma,tauinf,dt,maxT,F,minrho,maxrho,Nrho,nsat,destination,sinktype='statistical saturable')
	timeaxis,rhoaxis,ftmat=getrhotmat_thermionic(L,De,alpha,gamma,tauinf,.1,10,F,0.05,200,401,nsat,destination,sinktype='statistical saturable')
if 0:
	fluaxis=np.loadtxt("rhotmat0De0p601Dh0p2651_tsink0_401mesh_flupoints.txt")
	print("fluaxis: ",fluaxis)
	rho0points=np.loadtxt("rhotmat0De0p601Dh0p2651_tsink0_401mesh_rho0points.txt")
	print("rho0points: ",rho0points)
	timeaxis=np.loadtxt("rhotmat0De0p601Dh0p2651_tsink0_401mesh_timepoints.txt")*1000.
	ftmat= pd.read_csv("rhotmat0De0p601Dh0p2651_tsink0_401mesh.txt",fluaxis, delimiter = "\t",dtype=np.float64)
	print(ftmat)
	#timepoints=np.append(np.arange(-100,1000,20),np.arange(1000,8000,200))
	timepoints=np.append(np.arange(-100,1000,20),np.arange(1000,8200,100))
	powers =np.array([0.5,2.0])
	p=powers[0]
	scalefactor=np.array([0.77])
	s=0.77
	for p in powers:
		base="suemsim_nsat_2_"+str(p)+"mw"+str(s)+"_sf"
		print(p)
		print(base)
		if 1:
			#shifty=np.linspace(-142.865/2, 142.865/2, num=485)
			#shiftx=np.linspace(-217.391/2, 217.391/2, num=738)
			mag0=1200
			shiftx,shifty=ps.getpositionaxis(mag0)
			centercoord2=np.loadtxt("centercoord.txt")
			centercoord2_um=ps.coord2um(centercoord2,mag0,xpixel=738,ypixel=485)
			wx0s=wx0*s
			wy0s=wy0*s
			g1=103.5
			g2=21.14
			g3=38.33
			g1s=g1*s
			g2s=g2*s
			g3s=g3*s
			ephoton=3.73657e-19
			profilemat=1000*ps.getexposuremat(p,mag0,wx=wx0s,wy=wy0s,materialreflectivity=0,mux=0,muy=0,reprate=1e6,xpixel=738,ypixel=485,theta=0,radians=0, save=1,destination="beamprofile.txt")
			#ps.getexposuremat(p,mag0,wx=wx0,wy=wy0,mux=0,muy=0,reprate=1e6,xpixel=738,ypixel=485,theta=0, save=0,destination="beamprofile.txt")
			ps.plotprofilemat(profilemat,shiftx,shifty,save=1,show=1,autoclose=1,destination="profilemat_"+str(p)+"mW.tif")
			pround=np.round(np.amax(profilemat),3)
			print("maxprofilemat:",pround)
			print("fluenceinchamber:",ps.fluenceinchamber(p,materialreflectivity=0,reprate=1E6,wx=wx0s,wy=wy0s))
			suemsim(ftmat,profilemat,timepoints,fluaxis,timeaxis,destinationbase=base)
			fl_ss=ps.getfilenames(base, index=timepoints,extrastring="",delimiter="_t",suffix=".txt",exactmatch=1,verbose=1)
			sf_ss=ps.getstretchfactormulti(fl_ss,minimalstretch=1)
			ps.stretchdatamulti(fl_ss,sf_ss)
			fl_ss_stretched=ps.getfilenames(base, index=timepoints,extrastring="",delimiter="_t",suffix="_stretched.txt",exactmatch=1,verbose=1)
			ps.createimagemulti(fl_ss_stretched)
			ps.cropdatamulti(fl_ss_stretched,rx=738/2,ry=484/2,usemid=1,save=1,verbose=0,rotate=0)
			fl_ss_cropped=ps.getfilenames(base, index=timepoints,extrastring="",delimiter="_t",suffix="_stretched_cropped.txt",exactmatch=1,verbose=1)
			ps.createimagemulti(fl_ss_cropped)
if 0:
	fl_ss_stretched_tif=ps.getfilenames(base, index=timepoints,extrastring="",delimiter="_t",suffix="_stretched.tif",exactmatch=1,verbose=1)
	ps.addcaptionmulti(fl_ss_stretched_tif,base+"_cap",index=timepoints,prefix="_t",timepoints=timepoints,power=pround,mag=mag0,wd=None,av=None,timecaption=" ps",powercaption=" mJ/cm^2",wdcaption="WD =		 mm",avcaption="AV =	   KV",rgba=(200,0,0,255),integerindex=1)		
if 1:
	#simulaterhoseries_langmuir(L,De,K,tauinf,alpha,dt,maxT,F,rhoseries,nsat,destination,):
	time0,rhopoints0,rhotmat0=simulaterhoseries_langmuir(L,De,K,tauinf,alpha,.004,10,F,rhoseries,nsat,destination)
	plotrhoseries(time0,rhopoints0,rhotmat0,xmin=-0.5,xmax=10,destination="simlangmuirdiffpde0.tif")
	plotrhoseries(time0,rhopoints0,rhotmat0,xmin=-0.5,xmax=2,destination="simlangmuirdiffpde0_short.tif")
if 0:
	timepoints0=np.loadtxt("rhotmat0De0p601Dh0p2651_langmuirsink0_timepoints.txt")
	rhotmat0=np.loadtxt("rhotmat0De0p601Dh0p2651_langmuirsink0.txt")
	print(rhotmat0[:,0])
	timepoints_conv,rhot_conv=conv(timepoints0,rhotmat0[:,0],65)
	timepoints_conv,rhotmat_conv=convmulti(timepoints0,rhotmat0,65)
if 0:
	time0,rhopoints0,rhotmat0=simulaterhoseries_thermionicdecay(L,De,alpha,gamma,tauinf,.001,8,F,rhoseries,nsat,destination,sinktype='perfect thermionic')
	plotrhoseries(time0,rhopoints0,rhotmat0,destination="perfectdiffpde0.tif")
if 0:
	timepoints0=np.loadtxt("rhotmat0De0p601Dh0p2651_perfectsink6_timepoints.txt")
	rhotmat0=np.loadtxt("rhotmat0De0p601Dh0p2651_perfectsink6.txt")
	print(rhotmat0[:,0])
	timepoints_conv,rhot_conv=conv(timepoints0,rhotmat0[:,0],65)
	timepoints_conv,rhotmat_conv=convmulti(timepoints0,rhotmat0,65)
if 0:
	plt.plot(timepoints_conv,rhot_conv,'red')
	plt.plot(timepoints0,rhotmat0[:,0],'blue',linestyle='dashed')
	plt.xlabel('time (ns)')
	plt.ylabel('occupation $(\\times 10^{12})$')
	plt.savefig("saturablestatisticalsinkrho_sat_2rho_0_4.tif")
	plt.show(block=False)
	plt.pause(2)
	plt.close()
if 0:
	#plt.plot(timepoints_conv,rhot_conv,'red')
	plt.plot(timepoints_conv,np.transpose(rhotmat_conv),'blue',linestyle='dashed')
	plt.xlabel('time (ns)')
	plt.ylabel('occupation $(\\times 10^{12})$')
	plt.savefig("saturablestatisticalsinkrho_sat_2rho_0_4.tif")
	plt.show(block=False)
	plt.pause(2)
	plt.close()
if 0:
	utxn,ue_removed,u_n,rho_t, xn, timepoints, timeelasped=diffpde_saturable_langmuir(9.27,nsat,De,K,tauinf,alpha, L, dt,T, F,verbose=1,returnminimal=0)
	#utxn,ue_removed,u_n,rho_t, xn,timepoints, t1-t0
	np.savetxt("electronsstatsatsinkthermionicdecayrho_sat_2rho_0_4_2dmesh_utxn.txt",utxn)
	np.savetxt("electronsstatsatsinkthermionicdecayrho_sat_2rho_0_4_2dmesh_xn.txt",xn)
	np.savetxt("electronsstatsatsinkthermionicdecayrho_sat_2rho_0_4_2dmesh_timepoints.txt",timepoints)
	title="Electrons No Sink $\\rho_{e-}(0)=4\\times 10^{12}$, $\\rho_{sat}(0)=2\\times 10^{12}$"
	fig1, ax1 = plt.subplots(figsize=(3.8,	3.6))
	plt.tight_layout(w_pad=2, h_pad=2)
	Tmesh, X = np.meshgrid(timepoints, xn-xn[-1]/2)
	levels = np.logspace(-3, 0, 20)
	matplotlib.rc('font', family='serif', serif='cm10')
	matplotlib.rc('text', usetex=True)
	matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']
	rc('text', usetex=True)
	rc('font', weight='bold')
	#font = {'family' : 'arial','weight' : 'bold','size':18}
	#matplotlib.rc('font',**font)
	#matplotlib.rcParams.update({'font.size': 22})
	print(int(np.round(len(xn)/2)))
	print(len(xn))
	#colors = [plt.cm.nipy_spectral(i) for i in np.array([1.0,0.9,0.8,0.55,0.2,0.07,0])]
	#plt.margins(1)
	c=ax1.pcolormesh(xn[int(np.round(len(xn)/2)):-1]-xn[-1]/2,timepoints, utxn[:,int(np.round(len(xn)/2)):-1], norm=colors.PowerNorm(gamma=0.5),cmap='nipy_spectral', shading='auto')
	cbar = fig1.colorbar(c)
	ax1.set_ylabel('Time (ns)', fontsize='large', fontweight='bold',fontname="Arial")
	ax1.set_xlabel('Depth ($\\mu$m)', fontsize='large', fontweight='bold',fontname="Arial",labelpad=-2)
	ax1.xaxis.set_minor_locator(AutoMinorLocator())
	ax1.yaxis.set_minor_locator(AutoMinorLocator())
	cbar.set_label('Carrier Density, $\\rho_{e}$, ($\\times 10^{17} cm^{-3}$)',rotation=-90, fontsize='large', fontweight='bold',fontname="Arial", labelpad=12)
	plt.savefig("electronsstatsatsinkthermionicdecayrho_sat_2rho_0_4_2dmesh.tif")
	plt.show(block=False)
	plt.pause(10)
	plt.close()
if 1:
	utxn,ue_removed,u_n,rho_t, xn, timepoints, timeelasped=diffpde_saturable_langmuir(9.27,nsat,De,K,tauinf,alpha, L, dt,T, F,verbose=1,returnminimal=0)
	title="Trap Filling $\\rho_{sat}(0)=2\\times 10^{12}$"
	fig2, ax2 = plt.subplots()
	Tmesh, X = np.meshgrid(timepoints, rhopoints0)
	#colors = [plt.cm.nipy_spectral(i) for i in np.array([1.0,0.9,0.8,0.55,0.2,0.07,0])]
	c=ax2.pcolormesh(timepoints,rhopoints0, rhotmat0, norm=colors.PowerNorm(gamma=0.5),cmap='nipy_spectral', shading='auto')
	cbar = fig2.colorbar(c)
	plt.title(title)
	ax2.set_xlabel('time (ns)')
	ax2.set_ylabel('$\\rho_t (\\times 10^{12} cm^{-2})$')
	cbar.set_label('carrier density $(\\times 10^{12})/cm^2$')
	plt.savefig("rhotmat_perfect_rho_sat_2_2dmesh401.tif")
	plt.show(block=False)
	plt.pause(2)
	plt.close()
#I=np.exp(-(x-2.5)**2/1)
#u, utx,x, t, timeelasped=solver_BE(I,hnu,nsat,De, L, Nx, F, T,exportminimal=0,sinktype="perfect", user_action=None,verbose=1)

if 0:
	utxn,utxp,ue_removed,uh_removed,u_n,u_p,rho_t, xn,xp, timepoints, timeelasped=diffpde_saturable_statistical_sink(hnu,nsat,De,Dh, L, dt,  T,F,sinktype="statistical saturable",verbose=1,returnminimal=0)
	print(T)
	labels=("$\\rho_t$","$\\rho_{e-}$","$\\rho_{h+}$")
	title="1D Diffusion Saturable Statistical Sink $\\rho_{e-}(0)=4\\times 10^{12}$, $\\rho_{sat}(0)=2\\times 10^{12}$"
	plt.plot(timepoints,rho_t,'red',label=labels[0])
	plt.plot(timepoints,ue_removed,'green',linestyle='dotted',label=labels[1])
	plt.plot(timepoints,uh_removed,'blue',linestyle='dashed',label=labels[2])
	plt.legend(labels)
	plt.title(title)
	plt.xlabel('time (ns)')
	plt.ylabel('occupation $(\\times 10^{12})$')
	plt.savefig("saturablestatisticalsinkrho_sat_2rho_0_4.tif")
	plt.show(block=False)
	plt.pause(2)
	plt.close()
	title="Electrons Saturable Statistical Sink $\\rho_{e-}(0)=4\\times 10^{12}$, $\\rho_{sat}(0)=2\\times 10^{12}$"
	fig1, ax1 = plt.subplots()
	Tmesh, X = np.meshgrid(timepoints, xn-xn[-1]/2)
	levels = np.logspace(-3, 0, 20)
	#colors = [plt.cm.nipy_spectral(i) for i in np.array([1.0,0.9,0.8,0.55,0.2,0.07,0])]
	c=ax1.pcolormesh(xn-xn[-1]/2,timepoints, utxn, norm=colors.PowerNorm(gamma=0.5),cmap='nipy_spectral', shading='auto')
	cbar = fig1.colorbar(c)
	plt.title(title)
	ax1.set_ylabel('time (ns)')
	ax1.set_xlabel('position $(\\mu m)$')
	cbar.set_label('occupation $(\\times 10^{12})$')
	plt.savefig("electronssaturablestatisticalsinkrho_sat_2rho_0_4_2dmesh.tif")
	plt.show(block=False)
	plt.pause(2)
	plt.close()
	
	title="Holes Saturable Statistical Sink $\\rho_{e-}(0)=4\\times 10^{12}$, $\\rho_{sat}(0)=2\\times 10^{12}$"
	fig2, ax2 = plt.subplots()
	Tmesh, X = np.meshgrid(timepoints, xp-xp[-1]/2)
	levels = np.logspace(-3, 0, 20)
	#colors = [plt.cm.nipy_spectral(i) for i in np.array([1.0,0.9,0.8,0.55,0.2,0.07,0])]
	c=ax2.pcolormesh(xp-xp[-1]/2,timepoints, utxp, norm=colors.PowerNorm(gamma=0.5),cmap='nipy_spectral', shading='auto')
	cbar = fig2.colorbar(c)
	plt.title(title)
	ax1.set_ylabel('time (ns)')
	ax1.set_xlabel('position $(\\mu m)$')
	cbar.set_label('carrier density $(\\times 10^{17})/cm^3$')
	plt.savefig("holessaturablestatisticalinkrho_sat_2rho_0_4_2dmesh.tif")
	plt.show(block=False)
	plt.pause(2)
	plt.close()
if 0:
	utxn,utxp,ue_removed,uh_removed,u_n,u_p,rho_t, xn,xp, timepoints, timeelasped=diffpde_saturable_statistical_sink(hnu,nsat,De,Dh, L, dt,  T,F,sinktype="none",verbose=1,returnminimal=0)
	labels=("$\\rho_t$","$\\rho_{e-}$","$\\rho_{h+}$")
	title="1D Diffusion No Sink $\\rho_{e-}(0)=4\\times 10^{12}$"
	plt.plot(timepoints,rho_t,'red',label=labels[0])
	plt.plot(timepoints,ue_removed,'green',label=labels[1])
	plt.plot(timepoints,uh_removed,'blue',label=labels[2])
	plt.legend(labels)
	plt.title(title)
	plt.xlabel('time (ns)')
	plt.ylabel('occupation $(\\times 10^{12})$')
	plt.savefig("nosinkrho_sat_infrho_0_4.tif")
	plt.show(block=False)
	plt.pause(2)
	plt.close()
	
	title="Electrons No Sink $\\rho_{e-}(0)=4\\times 10^{12}$, $\\rho_{sat}(0)=2\\times 10^{12}$"
	fig1, ax1 = plt.subplots()
	Tmesh, X = np.meshgrid(timepoints, xn-xn[-1]/2)
	levels = np.logspace(-3, 0, 20)
	#colors = [plt.cm.nipy_spectral(i) for i in np.array([1.0,0.9,0.8,0.55,0.2,0.07,0])]
	c=ax1.pcolormesh(xn-xn[-1]/2,timepoints, utxn, norm=colors.PowerNorm(gamma=0.5),cmap='nipy_spectral', shading='auto')
	cbar = fig1.colorbar(c)
	plt.title(title)
	ax1.set_ylabel('time (ns)')
	ax1.set_xlabel('position $(\\mu m)$')
	cbar.set_label('carrier density $(\\times 10^{17})/cm^3$')
	plt.savefig("electronsnosinkrho_sat_2rho_0_4_2dmesh.tif")
	plt.show(block=False)
	plt.pause(2)
	plt.close()
	
	title="Holes No Sink $\\rho_{e-}(0)=4\\times 10^{12}$, $\\rho_{sat}(0)=2\\times 10^{12}$"
	fig2, ax2 = plt.subplots()
	Tmesh, X = np.meshgrid(timepoints, xp-xp[-1]/2)
	levels = np.logspace(-3, 0, 20)
	#colors = [plt.cm.nipy_spectral(i) for i in np.array([1.0,0.9,0.8,0.55,0.2,0.07,0])]
	c=ax2.pcolormesh(xp-xp[-1]/2,timepoints, utxp, norm=colors.PowerNorm(gamma=0.5),cmap='nipy_spectral', shading='auto')
	cbar = fig2.colorbar(c)
	plt.title(title)
	ax2.set_ylabel('time (ns)')
	ax2.set_xlabel('position $(\\mu m)$')
	cbar.set_label('carrier density $(\\times 10^{17})/cm^3$')
	plt.savefig("holesnosinkrho_sat_2rho_0_4_2dmesh.tif")
	plt.show(block=False)
	plt.pause(2)
	plt.close()
if 0:
	utxn,utxp,ue_removed,uh_removed,u_n,u_p,rho_t, xn,xp,timepoints, timeelasped=diffpde_saturable_statistical_sink(hnu,nsat,De,Dh, L, dt,  T,F,sinktype="perfect",verbose=1,returnminimal=0)
	labels=("$\\rho_t$","$\\rho_{e-}$","$\\rho_{h+}$")
	title="1D Diffusion Perfect Sink $\\rho_{e-}(0)=4\\times 10^{12}$"
	plt.plot(timepoints,rho_t,'red',label=labels[0])
	plt.plot(timepoints,ue_removed,'green',linestyle='dotted',label=labels[1])
	plt.plot(timepoints,uh_removed,'blue',linestyle='dashed',label=labels[2])
	plt.legend(labels)
	plt.title(title)
	plt.xlabel('time (ns)')
	plt.ylabel('occupation $(\\times 10^{12})$')
	plt.savefig("perfectsinkrho_sat_infrho_0_4.tif")
	plt.show(block=False)
	plt.pause(2)
	plt.close()
	fig1, ax1 = plt.subplots()
	Tmesh, X = np.meshgrid(timepoints, xp-xp[-1]/2)
	levels = np.logspace(-3, 0, 20)
	#colors = [plt.cm.nipy_spectral(i) for i in np.array([1.0,0.9,0.8,0.55,0.2,0.07,0])]
	
	title="Electrons Perfect Sink $\\rho_{e-}(0)=4\\times 10^{12}$, $\\rho_{sat}(0)=2\\times 10^{12}$"
	fig1, ax1 = plt.subplots()
	Tmesh, X = np.meshgrid(timepoints, xn-xn[-1]/2)
	levels = np.logspace(-3, 0, 20)
	#colors = [plt.cm.nipy_spectral(i) for i in np.array([1.0,0.9,0.8,0.55,0.2,0.07,0])]
	c=ax1.pcolormesh(xn-xn[-1]/2,timepoints, utxn, norm=colors.PowerNorm(gamma=0.5),cmap='nipy_spectral', shading='auto')
	cbar = fig1.colorbar(c)
	plt.title(title)
	ax1.set_ylabel('time (ns)')
	ax1.set_xlabel('position $(\\mu m)$')
	cbar.set_label('carrier density $(\\times 10^{17})/cm^3$')
	plt.savefig("electronsperfectsinkrho_sat_2rho_0_4_2dmesh.tif")
	plt.show(block=False)
	plt.pause(2)
	plt.close()
	
	title="Holes Perfect Sink $\\rho_{e-}(0)=4\\times 10^{12}$, $\\rho_{sat}(0)=2\\times 10^{12}$"
	fig2, ax2 = plt.subplots()
	Tmesh, X = np.meshgrid(timepoints, xp-xp[-1]/2)
	levels = np.logspace(-3, 0, 20)
	#colors = [plt.cm.nipy_spectral(i) for i in np.array([1.0,0.9,0.8,0.55,0.2,0.07,0])]
	c=ax2.pcolormesh(xp-xp[-1]/2,timepoints, utxp, norm=colors.PowerNorm(gamma=0.5),cmap='nipy_spectral', shading='auto')
	cbar = fig2.colorbar(c)
	plt.title(title)
	ax2.set_ylabel('time (ns)')
	ax2.set_xlabel('position $(\\mu m)$')
	cbar.set_label('carrier density $(\\times 10^{17})/cm^3$')
	plt.savefig("holesperfectsinkrho_sat_2rho_0_4_2dmesh.tif")
	plt.show(block=False)
	plt.pause(2)
	plt.close()
if 0:
	utxn,utxp,ue_removed,uh_removed,u_n,u_p,rho_t, xn,xp, timepoints, timeelasped=diffpde_saturable_statistical_sink(hnu,nsat,De,Dh, L, dt,  T,F,sinktype="perfect saturable",verbose=1,returnminimal=0)
	labels=("$\\rho_t$","$\\rho_{e-}$","$\\rho_{h+}$")
	title="1D Diffusion Saturable Perfect Sink $\\rho_{e-}(0)=4\\times 10^{12}$, $\\rho_{sat}(0)=2\\times 10^{12}$"
	plt.plot(timepoints,rho_t,'red',label=labels[0])
	plt.plot(timepoints,ue_removed,'green',linestyle='dotted',label=labels[1])
	plt.plot(timepoints,uh_removed,'blue',linestyle='dashed',label=labels[2])
	plt.legend(labels)
	plt.title(title)
	plt.xlabel('time (ns)')
	plt.ylabel('occupation $(\\times 10^{12})$')
	plt.savefig("saturableperfectsinkrho_sat_2rho_0_4.tif")
	plt.show(block=False)
	plt.pause(2)
	plt.close()
	
	title="Electrons Saturable Perfect Sink $\\rho_{e-}(0)=4\\times 10^{12}$, $\\rho_{sat}(0)=2\\times 10^{12}$"
	fig1, ax1 = plt.subplots()
	Tmesh, X = np.meshgrid(timepoints, xn-xn[-1]/2)
	levels = np.logspace(-3, 0, 20)
	#colors = [plt.cm.nipy_spectral(i) for i in np.array([1.0,0.9,0.8,0.55,0.2,0.07,0])]
	c=ax1.pcolormesh(xn-xn[-1]/2,timepoints, utxn, norm=colors.PowerNorm(gamma=0.5),cmap='nipy_spectral', shading='auto')
	cbar = fig1.colorbar(c)
	plt.title(title)
	ax1.set_ylabel('time (ns)')
	ax1.set_xlabel('position $(\\mu m)$')
	cbar.set_label('carrier density $(\\times 10^{17})/cm^3$')
	plt.savefig("electronssaturableperfectsinkrho_sat_2rho_0_4_2dmesh.tif")
	plt.show(block=False)
	plt.pause(2)
	plt.close()
	
	title="Holes Saturable Perfect Sink $\\rho_{e-}(0)=4\\times 10^{12}$, $\\rho_{sat}(0)=2\\times 10^{12}$"
	fig2, ax2 = plt.subplots()
	Tmesh, X = np.meshgrid(timepoints, xp-xp[-1]/2)
	levels = np.logspace(-3, 0, 20)
	#colors = [plt.cm.nipy_spectral(i) for i in np.array([1.0,0.9,0.8,0.55,0.2,0.07,0])]
	c=ax2.pcolormesh(xp-xp[-1]/2,timepoints, utxp, norm=colors.PowerNorm(gamma=0.5),cmap='nipy_spectral', shading='auto')
	cbar = fig2.colorbar(c)
	plt.title(title)
	ax2.set_ylabel('time (ns)')
	ax2.set_xlabel('position $(\\mu m)$')
	cbar.set_label('carrier density $(\\times 10^{17})/cm^3$')
	plt.savefig("holessaturableperfectsinkrho_sat_2rho_0_4_2dmesh.tif")
	plt.show(block=False)
	plt.pause(2)
	plt.close()