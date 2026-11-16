import pandas as pd

E = {
"Q001": '''The default is 1024. Increasing to 2048 is a good value to try. This just increases the soft file descriptor limit for ClearCase processes.''',
"Q002": '''Because of the code change in V10.1 GA, the off-line backup command could be blocked for short period of time while connect database command is executed.''',
"Q003": '''Action required for IBM Integration Bus Hypervisor Edition V9.0, WebSphere Message Broker Hypervisor Edition V8.0, V7.0 and IBM SOA Policy Pattern for Red Hat Enterprise Linux Server 2.0 for the GNU C library (glibc) vulnerabilities. [http://www.ibm.com/support/docview.wss?uid=swg21697721]''',
"Q004": '''One reason for these errors could be that the S-TAP Approval Needed is checked in the S-TAP Certification screen on the Appliance like this :- Uncheck the Approval box and the S-TAPs should start connecting .''',
"Q005": '''One reason for these errors could be that the S-TAP Approval Needed is checked in the S-TAP Certification screen on the Appliance like this :- Uncheck the Approval box and the S-TAPs should start connecting .''',
"Q006": '''In some other cases, the default value for nproc will be set to 1024, which may not allow to create enough threads in all processes；therefore, you have to check the ulimit setting on Linux.''',
"Q007": '''Alternatively, support for Traditional WebSphere for z/OS v9 and Liberty Profile for z/OS is available via the v7.2.0.0.14 Data Collector, which is available for download from the developerWorks''',
"Q008": '''Please review Support's Guide to IBM CORBA probes [attached] to understand the integration, alongside the probe's specific documentation.''',
"Q009": '''Even though the Role receives Traverse and Execute permissions for the Report Studio capability it will not be able to open Report Studio (this is only possible if you have Traverse and Execute permissions for the Professional User Interface Profile).''',
"Q010": '''RSTLICPGM/DLTLICPGM 5724C04 command failed to finish correctly and leaves the agent in an unknown state. DLTUSRPRF QAUTOMON and choose to remove all objects owned by this user.''',
"Q011": '''These component parameters are not automatically added as part of the database upgrade scripts. To enable the Enable Hold and Virtual Interview Multiple Response functionality you can manually create the component parameters.''',
"Q012": "NONE",
"Q013": '''DataPower MQ manager object's idle connection is not closed when using default cache timeout. Change the cache timeout on the IBM WebSphere DataPower MQ manager (mq-qm) object.''',
"Q014": '''To resolve the above error, set the Transaction handling>Isolation Level from none to Auto-Commit.''',
"Q015": '''The list of dbsets (Database Connections or Schema Repositories) displayed on Rational ClearQuest Web is controlled by the ClearQuest Maintenance Tool on the server where the Rational ClearQuest Web Server component exists.''',
"Q016": '''The following commands will reset the RBM's, ACL's, WebGUI and SSH services to the default settings.''',
"Q017": '''Refer to the SPSS Student Support Portal for information on who to contact to obtain your code for SPSS Statistics, Modeler or AMOS Student Version or Graduate Pack.''',
"Q018": '''CVE-2011-5320 CVE-2017-15670 CVE-2017-15671 CVE-2017-15804''',
"Q019": '''To generate an authorization code or license key for your SPSS product, access the "License Key Center".''',
"Q020": '''The Java heap settings are set in the config.sh script in the JAVA_OPTS line. For AIX, the default settings are way too low. Change the JAVA_OPTS="-Xms256m -Xmx256m" to a higher value like this: JAVA_OPTS="-Xms512m -Xms512m"''',
"Q021": '''The JBoss service is not available to run as a Windows service when configured with the ITCAM for J2EE agent/DC because this involves changes to the JBoss native connector files and this is currently not supported.''',
"Q022": "NONE",
"Q023": '''Select the *RC4* ciphers from the "Selected ciphers" list, and then click the "<<Remove" button.''',
"Q024": '''IBM Integration Bus Hypervisor Edition V9.0 and WebSphere Message Broker Hypervisor Edition V8.0 require customer action for security vulnerabilities in Red Hat (CVE-2016-0800, CVE-2016-0703, CVE-2016-0704,CVE-2016-1234)''',
"Q025": '''Yes, StoredIQ 7.6.0.5 and above support TLS 1.2.''',
}

qa = pd.read_csv("data/qa_v1.csv").fillna("")
qa["evidence_text"] = qa["qa_id"].map(E).fillna("")
qa.to_csv("data/qa_v1.csv", index=False)
print("Filled", (qa["evidence_text"] != "").sum(), "of", len(qa))
